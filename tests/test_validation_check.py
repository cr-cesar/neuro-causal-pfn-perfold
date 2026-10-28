"""Validation gate: ranking agreement and noise on synthetic replica results."""
import numpy as np
import pandas as pd
import pytest

from ncpfold.validation_check import check, load_side, main, parse_sources

DEFICITS = [f"d{i}" for i in range(16)]


def _write(root, classifier, learner, levels, seeds=(0,), noise=0.0, rng=None):
    """``replica_results.csv`` per seed: pehe_paper = level of the deficit (+ noise), 2 folds."""
    rng = rng or np.random.default_rng(0)
    for s in seeds:
        rows = [{"representation": f"E1|E1|seed{s}|chain|primary|grp-giles-extras",
                 "deficit": d, "fold": k, "classifier": classifier, "learner": learner,
                 "pehe_paper": levels[j] + noise * rng.normal()}
                for j, d in enumerate(DEFICITS) for k in range(2)]
        out = root / "replica" / "E1" / "E1" / f"seed{s}" / f"{classifier}-{learner}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "replica_results.csv", index=False)


def _side(tmp_path, name, offsets, shuffle=None):
    """Five estimators whose mean PEHE follows ``offsets`` (lower is better)."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.5, size=len(DEFICITS))
    roots = {}
    cells = [("logistic_regression", "s_learner"), ("extra_trees", "t_learner"),
             ("causalpfn", "in_context"), ("pfn", "in_context"), ("tabpfn", "s_learner")]
    for (c, l), off in zip(cells, offsets):
        tag = "lr" if c in ("logistic_regression", "extra_trees") else c
        root = tmp_path / name / tag
        _write(root, c, l, base + off + (shuffle(rng) if shuffle else 0.0), seeds=(0, 1, 2))
        roots[tag] = str(root)
    return [(t, r) for t, r in roots.items()]


def test_same_order_passes_and_reversed_order_fails(tmp_path):
    offsets = [0.00, 0.05, 0.10, 0.15, 0.20]
    val = load_side(_side(tmp_path, "val", offsets))
    same = load_side(_side(tmp_path, "test_same", [o * 0.8 for o in offsets]))
    rev = load_side(_side(tmp_path, "test_rev", offsets[::-1]))
    ok = check(val, same, "receptor", builtin="all", n_boot=200)
    assert ok[0]["kind"] == "ranking" and ok[0]["n_estimators"] == 5 and ok[0]["n_deficits"] == 16
    assert ok[0]["rho"] == pytest.approx(1.0) and ok[0]["passes"]
    bad = check(val, rev, "receptor", builtin="all", n_boot=200)
    assert bad[0]["rho"] == pytest.approx(-1.0) and not bad[0]["passes"]


def test_builtin_best_keeps_the_validation_winner_only(tmp_path):
    offsets = [0.00, 0.05, 0.10, 0.15, 0.20]
    val = load_side(_side(tmp_path, "val", offsets))
    test = load_side(_side(tmp_path, "test", offsets))
    rows = check(val, test, "genetics", builtin="best", n_boot=100)
    names = [r["estimator"] for r in rows[1:]]
    assert "logistic_regression/s_learner" in names and "extra_trees/t_learner" not in names
    assert any(n.startswith("pfn:") for n in names) and rows[0]["n_estimators"] == 4


def test_noise_flags_a_seed_spread_over_002(tmp_path):
    rng = np.random.default_rng(3)
    base = rng.uniform(0.2, 0.5, size=len(DEFICITS))
    val_sources, test_sources = [], []
    for i, (c, l) in enumerate([("logistic_regression", "s_learner"), ("causalpfn", "in_context"),
                                ("pfn", "in_context")]):
        tag = "lr" if c == "logistic_regression" else c
        for side, srcs in (("val", val_sources), ("test", test_sources)):
            root = tmp_path / side / tag
            if c == "pfn" and side == "val":
                for s, shift in zip((0, 1, 2), (-0.05, 0.0, 0.05)):   # seed SD ~0.04
                    _write(root, c, l, base + 0.1 * i + shift, seeds=(s,))
            else:
                _write(root, c, l, base + 0.1 * i, seeds=(0, 1, 2))
            srcs.append((tag, str(root)))
    rows = check(load_side(val_sources), load_side(test_sources), "receptor", n_boot=100)
    pfn = next(r for r in rows[1:] if r["estimator"].startswith("pfn:"))
    assert pfn["val_seed_sd"] == pytest.approx(np.std([-0.05, 0.0, 0.05]))
    assert pfn["noise_over"] and rows[0]["noise_over"]
    lr = next(r for r in rows[1:] if r["estimator"].startswith("logistic"))
    assert lr["val_seed_sd"] == pytest.approx(0.0) and not lr["noise_over"]


def test_paired_noise_flags_a_difference_that_swings_across_deficits(tmp_path):
    rng = np.random.default_rng(4)
    base = rng.uniform(0.2, 0.5, size=len(DEFICITS))
    swing = np.where(np.arange(len(DEFICITS)) % 2 == 0, -0.12, 0.12)   # mean 0.05 behind, SD 0.12
    sources = {"val": [], "test": []}
    for side in sources:
        for tag, c, l, levels in (("lr", "logistic_regression", "s_learner", base),
                                  ("causalpfn", "causalpfn", "in_context", base + 0.05 + swing),
                                  ("pfn", "pfn", "in_context", base + 0.10)):
            root = tmp_path / side / tag
            _write(root, c, l, levels)
            sources[side].append((tag, str(root)))
    rows = check(load_side(sources["val"]), load_side(sources["test"]), "receptor", n_boot=500)
    by = {r["estimator"].split(":")[0]: r for r in rows[1:]}
    assert by["causalpfn"]["val_boot_sd"] > 0.02 and by["causalpfn"]["noise_over"]
    assert by["pfn"]["val_boot_sd"] == pytest.approx(0.0, abs=1e-12) and not by["pfn"]["noise_over"]


def test_cli_writes_one_split_and_keeps_the_other(tmp_path):
    offsets = [0.00, 0.05, 0.10, 0.15, 0.20]
    val = _side(tmp_path, "val", offsets)
    test = _side(tmp_path, "test", offsets)
    out = tmp_path / "status" / "validation_check.csv"
    args = sum([["--val", f"{t}={r}"] for t, r in val], []) + sum([["--test", f"{t}={r}"] for t, r in test], [])
    main(["--split", "receptor", "--n-boot", "50", "--out", str(out)] + args)
    main(["--split", "genetics", "--n-boot", "50", "--out", str(out)] + args)
    main(["--split", "receptor", "--n-boot", "50", "--out", str(out)] + args)
    df = pd.read_csv(out)
    assert sorted(df.loc[df["kind"] == "ranking", "split"]) == ["genetics", "receptor"]
    assert (df["split"] == "receptor").sum() == 5     # ranking + 4 estimators, not duplicated


def test_a_root_under_two_tags_is_refused(tmp_path):
    root = tmp_path / "base"
    _write(root, "logistic_regression", "s_learner", np.full(len(DEFICITS), 0.3))
    _write(root, "causalpfn", "in_context", np.full(len(DEFICITS), 0.35))
    with pytest.raises(SystemExit, match="more than one tag"):
        load_side([("lr", str(root)), ("causalpfn", str(root) + "/")])
    one = load_side([("base", str(root))])
    assert sorted(one["estimator"].unique()) == ["base:causalpfn/in_context", "logistic_regression/s_learner"]


def test_two_scorings_on_one_side_are_refused(tmp_path):
    root = tmp_path / "val"
    _write(root, "logistic_regression", "s_learner", np.full(len(DEFICITS), 0.3))
    old = root / "replica" / "E1" / "E1" / "seed0" / "old"
    old.mkdir(parents=True)
    pd.DataFrame([{"representation": "E1|E1|seed0|chain|primary|grp-giles", "deficit": d, "fold": 0,
                   "classifier": "pfn", "learner": "in_context", "pehe_paper": 0.4}
                  for d in DEFICITS]).to_csv(old / "replica_results.csv", index=False)
    with pytest.raises(SystemExit, match="several scorings"):
        load_side([("val", str(root))])


def test_sources_need_tags():
    assert parse_sources(["a=x/y"]) == [("a", "x/y")]
    with pytest.raises(ValueError):
        parse_sources(["x/y"])
