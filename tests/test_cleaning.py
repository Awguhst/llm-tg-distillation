"""
Tests of the polymer keys and the cleaning rules in pipeline/common/cleaning.py.
Run with:  pytest   (from the project root; pyproject.toml puts pipeline/ on the path)
"""
import pandas as pd
import pytest

from common.cleaning import KeySet, canonical_smiles, merge_duplicates, polymer_id, polymer_keys, ring_closure_key


def test_same_chain_cut_elsewhere_matches_on_the_ring_key():
    # poly(ethylene oxide) written from two different backbone atoms
    assert canonical_smiles("*CCO*") != canonical_smiles("*COC*")
    assert ring_closure_key("*CCO*") == ring_closure_key("*COC*")
    assert polymer_id("*CCO*") == polymer_id("*COC*")


def test_vinyl_unit_has_no_ring_key_but_an_exact_key():
    # polystyrene: the two * sit on bonded atoms, so no ring can be closed; the exact SMILES is
    # already cut-independent for such units
    exact, ring = polymer_keys("*CC(*)c1ccccc1")
    assert exact == canonical_smiles("*CC(*)c1ccccc1")
    assert ring is None
    assert polymer_id("*CC(*)c1ccccc1") == exact


def test_one_atom_repeat_unit_has_no_ring_key():
    assert ring_closure_key("*C*") is None


def test_multi_fragment_smiles_has_no_ring_key():
    # a repeat unit accompanied by a small molecule: two fragments separated by "."
    assert ring_closure_key("*CCO*.O") is None
    assert ring_closure_key("*CC.CC*") is None


def test_repeat_unit_written_twice_is_not_matched():
    # Documented limitation: neither key recognizes *CCCC* as the same polymer as *CC*.
    assert canonical_smiles("*CCCC*") != canonical_smiles("*CC*")
    assert ring_closure_key("*CCCC*") != ring_closure_key("*CC*")
    assert polymer_id("*CCCC*") != polymer_id("*CC*")


def test_stereo_smiles_parse_and_get_keys():
    # a polydiene unit with a double-bond configuration; RDKit keeps the stereo mark
    exact, ring = polymer_keys("*C/C=C/C*")
    assert exact is not None and "/" in exact
    assert ring is not None
    # a tetrahedral centre
    exact, ring = polymer_keys("*C[C@H](C)O*")
    assert exact is not None and "@" in exact
    assert ring is not None


def test_unparsable_and_empty_smiles_give_none():
    assert canonical_smiles("not a smiles") is None
    assert canonical_smiles("") is None
    assert canonical_smiles(None) is None
    assert polymer_keys("C1CC") == (None, None)


def test_keyset_match_types():
    keys = KeySet(["*CCO*", "*CC(*)c1ccccc1"])
    assert keys.match_type("*CCO*") == "exact"
    assert keys.match_type("*COC*") == "ring"          # same polymer, different cut
    assert keys.match_type("*CC(*)c1ccccc1") == "exact"
    assert keys.match_type("*CC(*)c1ccc(C)cc1") is None
    assert keys.match_type("*CCCCO*") is None
    matches = keys.matches(pd.Series(["*COC*", "*CCCCO*", "*CC(*)c1ccccc1"]))
    assert list(matches) == [True, False, True]


def test_merge_duplicates_takes_the_median_over_either_key():
    df = pd.DataFrame({"smiles": ["*CCO*", "*COC*", "*OCC*", "*CC(*)c1ccccc1"], "tg_celsius": [-60.0, -70.0, -50.0, 100.0]})
    merged = merge_duplicates(df)
    assert len(merged) == 2
    peo = merged[merged["smiles"] != canonical_smiles("*CC(*)c1ccccc1")].iloc[0]
    assert peo["tg_celsius"] == pytest.approx(-60.0)   # median of -70, -60, -50
