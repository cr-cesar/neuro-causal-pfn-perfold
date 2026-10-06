import json
import os

import numpy as np
import pytest

from ncpfold.folds import all_folds
from ncpfold.score import fold_lookup, rep_meta


def _write_rep(rep_dir, n, n_folds, dim=4, seed=0):
    os.makedirs(rep_dir, exist_ok=True)
    Z = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    for k, (tr, te) in enumerate(all_folds(n, n_folds)):
        np.savez(os.path.join(rep_dir, f"fold{k}.npz"), Ztr=Z[tr], Zte=Z[te], tr_idx=tr, te_idx=te,
                 files=np.array([f"f{i}" for i in range(n)]), fold=k, n_folds=n_folds, dim=dim,
                 eid=np.array("E9"), label=np.array("E9"), seed=seed, budget=np.array("chain"),
                 task=np.array("disconnectome"))
    with open(os.path.join(rep_dir, "meta.json"), "w") as f:
        json.dump({"eid": "E9", "label": "E9", "seed": seed, "task": "disconnectome",
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


def test_singles_mask_and_restricted_lookup(tmp_path):
    from ncpfold.score import restrict_test, singles_mask

    n, n_folds = 37, 5
    Z = _write_rep(str(tmp_path / "rep"), n, n_folds)
    names = [f"f{i}" for i in range(n)]
    # images 3 and 4 are one patient, 10 and 11 another; everyone else is single
    groups = {"f3": ("p1", 0), "f4": ("p1", 1), "f10": ("p2", 0), "f11": ("p2", 1)}
    mask = singles_mask(names, groups)
    assert mask.sum() == n - 4 and not mask[[3, 4, 10, 11]].any()
    assert singles_mask(names, None).all()

    lookup = fold_lookup(str(tmp_path / "rep"), n)
    for (tr, te), (tr2, te2) in zip(all_folds(n, n_folds), restrict_test(all_folds(n, n_folds), mask)):
        assert np.array_equal(tr, tr2) and set(te2) <= set(te) and not set(te2) & {3, 4, 10, 11}
        Ztr, Zte = lookup(tr2, te2)
        assert np.array_equal(Ztr, Z[tr]) and np.array_equal(Zte, Z[te2])
    # a test set that is not a subset of one stored fold is refused
    tr, te = all_folds(n, n_folds)[0]
    other = all_folds(n, n_folds)[1][1]
    with pytest.raises(SystemExit):
        lookup(tr, np.concatenate([te[:2], other[:2]]))


def test_seed_ensemble_concatenates_fold_by_fold(tmp_path):
    from ncpfold.score import ensemble_lookup, ensemble_units

    n, n_folds = 37, 5
    dirs = [str(tmp_path / f"seed{s}" / "folds") for s in range(3)]
    Zs = [_write_rep(d, n, n_folds, seed=s) for s, d in enumerate(dirs)]
    units = ensemble_units(dirs, ensemble=True, ensemble_only=False)
    assert len(units) == 4 and units[-1][0]["label"] == "E9+x3" and units[-1][0]["seeds"] == [0, 1, 2]
    assert [u[0]["label"] for u in ensemble_units(dirs, True, True)] == ["E9+x3"]
    assert len(ensemble_units(dirs[:1], True, False)) == 1       # a single seed has no ensemble
    lookup = ensemble_lookup([fold_lookup(d, n) for d in dirs])
    assert lookup.n_folds == n_folds
    for tr, te in all_folds(n, n_folds):
        Ztr, Zte = lookup(tr, te)
        assert Ztr.shape == (len(tr), 12) and Zte.shape == (len(te), 12)
        assert np.allclose(Ztr, np.concatenate([Z[tr] for Z in Zs], axis=1))
        assert np.allclose(Zte, np.concatenate([Z[te] for Z in Zs], axis=1))
