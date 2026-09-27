"""
CPU smoke test of the MoLFormer training code: 20 polymers, 1 epoch, patience 1.
Marked slow (it downloads or loads the 45-million-parameter checkpoint and takes about a minute
on a CPU) and therefore skipped by default; run it with  pytest -m slow
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

POLYMERS = pd.DataFrame({
    "smiles": ["*CC*", "*CC(*)C", "*CC(*)c1ccccc1", "*CCO*", "*CCCCO*", "*CC(*)Cl", "*CC(*)C#N", "*CC(*)OC(C)=O",
               "*CC(C)(*)C(=O)OC", "*CC(*)C(=O)OC", "*C(=O)CCCCCN*", "*OC(=O)c1ccc(cc1)C(=O)OCC*", "*C(=O)c1ccc(cc1)C(=O)Nc1ccc(cc1)N*",
               "*C(F)(F)C(F)(F)*", "*CC(*)F", "*O[Si](C)(C)*", "*CC(*)c1ccncc1", "*CC(*)C(=O)N", "*OC(=O)OCC*", "*CC(*)c1ccc(C)cc1"],
    "tg_celsius": [-120.0, -10.0, 100.0, -60.0, -80.0, 80.0, 100.0, 30.0, 105.0, 10.0, 50.0, 70.0, 250.0, 120.0, 40.0, -125.0, 140.0, 165.0, 10.0, 110.0],
})


def test_one_epoch_on_cpu(monkeypatch):
    from common import molformer

    monkeypatch.setattr(molformer, "DEVICE", "cpu")
    try:
        molformer.get_tokenizer()
    except OSError as error:               # no cached checkpoint and no network
        pytest.skip(f"MoLFormer checkpoint not available: {error}")
    train, val, test = POLYMERS.iloc[:14], POLYMERS.iloc[14:17], POLYMERS.iloc[17:]
    mean, std = float(train["tg_celsius"].mean()), float(train["tg_celsius"].std())

    molformer.set_seed(0)
    model = molformer.load_model()
    history = molformer.fit(model, train, val, mean, std, max_epochs=1, patience=1, test=test, verbose=False)
    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_mse_standardized", "val_mae", "test_mae"}
    assert np.isfinite(history[0]["val_mae"]) and np.isfinite(history[0]["test_mae"])

    predictions = molformer.predict(model, test["smiles"], mean, std)
    assert predictions.shape == (len(test),)
    assert np.all(np.isfinite(predictions))
    metrics = molformer.compute_metrics(test["tg_celsius"].values, predictions)
    assert set(metrics) == {"mae", "rmse", "r2", "spearman"}
