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


def test_lookup_in_bracketed_variant_dir(tmp_path):
    # variant labels such as E3[backbone=cnn] contain glob metacharacters
    n, n_folds = 37, 5
    rep = str(tmp_path / "E3[backbone=cnn]" / "seed0" / "folds")
    Z = _write_rep(rep, n, n_folds)
    lookup = fold_lookup(rep, n)
    assert lookup.n_folds == n_folds and rep_meta(rep)["eid"] == "E9"
    tr, te = all_folds(n, n_folds)[0]
    assert np.allclose(lookup(tr, te)[1], Z[te])


def test_per_fold_estimator_dispatches_on_the_fold_the_lookup_served(tmp_path, monkeypatch):
    import ncpfold.score as sc

    n, n_folds = 30, 3
    _write_rep(str(tmp_path / "rep"), n, n_folds)
    lookup = fold_lookup(str(tmp_path / "rep"), n)
    seen = []

    class _FakePFN:
        def __init__(self, path, **kw):
            self.path = path

        def __call__(self, Xtr, W, Y, Xte):
            seen.append(self.path)
            return np.full(len(Xte), 0.7), np.full(len(Xte), 0.2)

    import neurocausalpfn.pfn.estimator as est
    monkeypatch.setattr(est, "PFNEstimator", _FakePFN)
    for k in range(n_folds):
        (tmp_path / f"run_fold{k}").mkdir()
        (tmp_path / f"run_fold{k}" / "pfn.pt").write_bytes(b"x")
    pfe = sc.PerFoldEstimator(str(tmp_path / "run_fold{fold}" / "pfn.pt"))
    folds = all_folds(n, n_folds)
    for k in (2, 0, 1):                                    # any order: the lookup sets the fold
        tr, te = folds[k]
        lookup(tr, te)
        p1, p0 = pfe(None, None, None, np.zeros((len(te), 4)))
        assert seen[-1].endswith(f"run_fold{k}/pfn.pt") and p1.shape == (len(te),)
    # a restricted test side (singles) still resolves to its stored fold
    tr, te = folds[1]
    lookup(tr, te[::2])
    pfe(None, None, None, np.zeros((len(te[::2]), 4)))
    assert seen[-1].endswith("run_fold1/pfn.pt")
    with pytest.raises(ValueError):
        sc.PerFoldEstimator("no_placeholder.pt")
    sc.CURRENT["fold"] = -1
    with pytest.raises(RuntimeError):
        pfe(None, None, None, np.zeros((2, 4)))


def test_resolve_classifiers_registers_the_panel_and_keeps_the_estimator_order(monkeypatch):
    import ncpfold.score as sc
    from neurocausalpfn.prior import giles_replica as gr

    registered = []
    import neurocausalpfn.pfn.panel as panel
    monkeypatch.setattr(panel, "register_panel",
                        lambda names, **kw: registered.extend(names) or {})
    monkeypatch.setattr(gr, "register_estimator", lambda name, fn: None)
    assert sc.resolve_classifiers(None) == ["logistic_regression", "extra_trees"] and not registered
    assert sc.resolve_classifiers(["tabpfn", "tabicl"]) == ["tabpfn", "tabicl"]
    assert registered == ["tabpfn", "tabicl"]
    monkeypatch.setattr(sc, "PerFoldEstimator", lambda path: path)
    assert sc.resolve_classifiers(["extra_trees"], ["run/fold{fold}/pfn.pt"]) == ["extra_trees", "pfn"]
    assert sc.resolve_classifiers(["tabicl"], ["run/fold{fold}/pfn.pt"], only_estimators=True) == ["pfn"]
    with pytest.raises(ValueError):
        sc.resolve_classifiers(None, ["something_else"])


def test_validation_extras_are_served_from_the_training_side(tmp_path):
    """--test-extras: the never-tested images form pseudo-folds; the lookup
    serves both sides from the stored fold's Ztr, each extra tested once."""
    from ncpfold.folds import GroupTable
    from ncpfold.score import extras_folds, extras_mask

    n, n_folds = 40, 4
    names = [f"img{i:02d}" for i in range(n)]
    # groups: images 0-29 single rank-0 patients; 30-37 later acquisitions of patients 0-7; 38-39 a group without rank 0
    groups = GroupTable({names[i]: (f"g{i}", 0) for i in range(30)})
    groups.update({names[30 + j]: (f"g{j}", 1) for j in range(8)})
    groups.update({names[38]: ("gx", 1), names[39]: ("gx", 2)})
    mask = extras_mask(names, groups)
    assert mask.sum() == 10 and mask[30:].all() and not mask[:30].any()
    # stored folds: KFold over the 30 primaries, extras train everywhere (giles mode)
    folds = all_folds(n, n_folds, names=names, groups=groups, mode="giles")
    rep = str(tmp_path / "rep"); os.makedirs(rep)
    Z = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    for k, (tr, te) in enumerate(folds):
        np.savez(os.path.join(rep, f"fold{k}.npz"), Ztr=Z[tr], Zte=Z[te], tr_idx=tr, te_idx=te,
                 files=np.array(names), fold=k, n_folds=n_folds, dim=3)
    vfolds = extras_folds(folds, mask, names)
    tested = np.concatenate([te for _, te in vfolds])
    assert sorted(tested.tolist()) == list(range(30, 40))             # every extra once
    for (tr, te), (tr0, te0) in zip(vfolds, folds):
        assert not (set(te.tolist()) & set(te0.tolist()))               # no benchmark test image is a query
        assert set(te.tolist()) <= set(tr0.tolist()) and set(tr.tolist()) <= set(tr0.tolist())
    lookup = fold_lookup(rep, n)
    for tr, te in vfolds:
        if len(te) == 0:
            continue
        Ztr, Zte = lookup(tr, te)
        np.testing.assert_array_equal(Ztr, Z[tr]); np.testing.assert_array_equal(Zte, Z[te])
    with pytest.raises(ValueError):
        extras_mask(names, None)
    # an exported fold table marks the extras with fold -1
    gt = GroupTable(groups); gt.folds = {nm: (-1 if mask[i] else i % n_folds) for i, nm in enumerate(names)}
    np.testing.assert_array_equal(extras_mask(names, gt), mask)


def test_seed_ensemble_of_pfn_checkpoints_averages_potential_outcomes(monkeypatch, tmp_path):
    """'a.pt+b.pt' registers one estimator whose mu1 / mu0 are the members' means."""
    import ncpfold.score as sc

    class _Fake:
        def __init__(self, path, **kw):
            self.v = float(os.path.basename(path)[1])
        def __call__(self, Xtr, W, Y, Xte):
            n = len(Xte)
            return np.full(n, self.v), np.full(n, -self.v)

    import neurocausalpfn.pfn.estimator as est
    monkeypatch.setattr(est, "PFNEstimator", _Fake)
    e = sc.make_pfn_estimator("p1.pt+p3.pt")
    assert isinstance(e, sc.EnsembleEstimator) and len(e.members) == 2
    mu1, mu0 = e(None, None, None, np.zeros((4, 2)))
    np.testing.assert_allclose(mu1, 2.0); np.testing.assert_allclose(mu0, -2.0)
    assert isinstance(sc.make_pfn_estimator("p2.pt"), _Fake)
    e2 = sc.make_pfn_estimator("outputs/a/fold{fold}/pfn.pt+outputs/b/fold{fold}/pfn.pt")
    assert all(isinstance(m, sc.PerFoldEstimator) for m in e2.members)
