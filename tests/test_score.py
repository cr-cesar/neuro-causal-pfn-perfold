import json
import os

import numpy as np
import pytest

from ncpfold.folds import all_folds
from ncpfold.score import fold_lookup, rep_meta


def _write_rep(rep_dir, n, n_folds, dim=4):
    os.makedirs(rep_dir, exist_ok=True)
    Z = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    for k, (tr, te) in enumerate(all_folds(n, n_folds)):
        np.savez(os.path.join(rep_dir, f"fold{k}.npz"), Ztr=Z[tr], Zte=Z[te], tr_idx=tr, te_idx=te,
                 files=np.array([f"f{i}" for i in range(n)]), fold=k, n_folds=n_folds, dim=dim,
                 eid=np.array("E9"), label=np.array("E9"), seed=0, budget=np.array("chain"),
                 task=np.array("disconnectome"))
    with open(os.path.join(rep_dir, "meta.json"), "w") as f:
        json.dump({"eid": "E9", "label": "E9", "seed": 0, "task": "disconnectome",
                   "n_folds": n_folds, "budget": "chain", "dim": dim}, f)
    return Z


def test_lookup_matches_replica_folds(tmp_path):
    n, n_folds = 37, 5
    Z = _write_rep(str(tmp_path / "rep"), n, n_folds)
    lookup = fold_lookup(str(tmp_path / "rep"), n)
    assert lookup.n_folds == n_folds
    for tr, te in all_folds(n, n_folds):
        Ztr, Zte = lookup(tr, te)
        assert np.allclose(Ztr, Z[tr]) and np.allclose(Zte, Z[te])
    meta = rep_meta(str(tmp_path / "rep"))
    assert meta["eid"] == "E9" and meta["n_folds"] == n_folds


def test_lookup_rejects_other_splits(tmp_path):
    n = 37
    _write_rep(str(tmp_path / "rep"), n, 5)
    lookup = fold_lookup(str(tmp_path / "rep"), n)
    tr, te = all_folds(n, 4)[0]                       # different fold count -> no match
    with pytest.raises(SystemExit):
        lookup(tr, te)
    with pytest.raises(SystemExit):
        fold_lookup(str(tmp_path / "rep"), n + 1)      # listing size differs
