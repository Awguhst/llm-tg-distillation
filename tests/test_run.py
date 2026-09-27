"""Tests of pipeline/run.py: the step table is consistent, and --list / --dry-run run nothing."""
import os
import subprocess
import sys

import run


def test_every_step_names_existing_scripts():
    for name, scripts, inputs, outputs, only_if_missing in run.STEPS:
        assert scripts and inputs is not None and outputs, name
        for script in scripts:
            assert os.path.exists(script), f"{name}: {script} does not exist"


def test_steps_are_in_dependency_order():
    # every input of a step is produced by an earlier step, or is an original download (step "download")
    produced = set()
    for name, scripts, inputs, outputs, only_if_missing in run.STEPS:
        for f in inputs:
            assert f in produced, f"{name} needs {os.path.basename(f)} before any step writes it"
        produced.update(outputs)


def test_status_logic(tmp_path):
    a, b = tmp_path / "in.txt", tmp_path / "out.txt"
    assert run.status([str(a)], [str(b)], False) == "missing inputs"
    a.write_text("x")
    assert run.status([str(a)], [str(b)], False) == "missing outputs"
    b.write_text("y")
    os.utime(str(a), (2_000_000_000, 2_000_000_000))    # input newer than output
    assert run.status([str(a)], [str(b)], False) == "inputs newer"
    assert run.status([str(a)], [str(b)], True) == "up to date"     # training steps ignore timestamps


def test_list_and_dry_run_execute_nothing():
    script = os.path.join(os.path.dirname(run.__file__), "run.py")
    out = subprocess.run([sys.executable, script, "--list"], capture_output=True, text=True, check=True).stdout
    assert "figures" in out and "report" in out
    out = subprocess.run([sys.executable, script, "--dry-run", "figures"], capture_output=True, text=True, check=True).stdout
    assert "done in" not in out
    bad = subprocess.run([sys.executable, script, "no_such_step"], capture_output=True, text=True)
    assert bad.returncode != 0 and "unknown step" in bad.stderr
