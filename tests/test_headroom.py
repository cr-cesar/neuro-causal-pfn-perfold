"""Step-0 oracles: truth from the replica's labels, and the per-cell oracle
and probe numbers on synthetic arrays."""
import numpy as np
import pandas as pd

from ncpfold.headroom import oracle_rows, truth_from_labels


def test_truth_from_labels_maps_hits_to_tau():
    df = pd.DataFrame({"motor_W0": [1, 0, 1, 0], "motor_W1": [0, 1, 1, 0]})
    mask, tau = truth_from_labels(df, "motor")
    assert mask.tolist() == [True, True, True, False]
    assert tau[:3].tolist() == [-1.0, 1.0, 0.0]


def test_oracles_recover_a_linear_truth_and_probe_the_hits():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    Z = rng.normal(size=(n, d))
    score = Z[:, 0] - Z[:, 1]
    tau = np.where(score > 0.8, 1.0, np.where(score < -0.8, -1.0, 0.0))
    A = np.column_stack([np.clip(0.05 + 0.1 * Z[:, 0], 0, 1), np.clip(0.05 + 0.1 * Z[:, 1], 0, 1)])
    hits = {"W1": (tau == 1).astype(int), "W0": (tau == -1).astype(int)}
    groups = np.arange(n) // 2
    tr, te = np.arange(0, 300), np.arange(300, 400)
    r = oracle_rows(Z[tr], Z[te], A[tr], A[te], tau[tr], tau[te],
                    {k: v[tr] for k, v in hits.items()}, groups[tr])
    assert r["n_train"] == 300 and r["n_test"] == 100 and 0 < r["share_zero_test"] < 1
    assert r["pehe_latent_hgb"] < 0.6 and r["pehe_latent_ridge"] < 0.8 and r["pehe_anatomy_hgb"] < 0.7
    assert r["auc_probe_W1"] > 0.9 and r["auc_probe_W0"] > 0.9
    # too few rows: only the counts
    small = oracle_rows(Z[:5], Z[te], None, None, tau[:5], tau[te], {}, groups[:5])
    assert set(small) == {"n_train", "n_test", "share_zero_test"}


def test_probe_is_nan_when_a_split_holds_one_class():
    rng = np.random.default_rng(1)
    n = 60
    Z = rng.normal(size=(n, 4)); tau = rng.choice([-1.0, 0.0, 1.0], size=n)
    hits = {"W1": np.r_[np.ones(58, int), 0, 0]}        # two negatives only: some folds see one class
    groups = np.arange(n)
    r = oracle_rows(Z[:50], Z[50:], None, None, tau[:50], tau[50:], {"W1": hits["W1"][:50]}, groups[:50])
    assert "pehe_latent_hgb" in r and np.isnan(r["auc_probe_W1"])
