"""Step 1 of the 27 Oct plan: does the validation set rank the estimators as the test set does?

The validation set (``--test-extras``: the never-tested images, in pseudo-folds
by patient) may select among trained variants only if it orders the estimators
as the 2,828 test images do. Per split, from scorings that already exist:

- ranking: Spearman correlation between the estimators' mean paper-scale PEHE
  in validation and on the test set, with a bootstrap over the deficits (the
  same resampled deficits on both sides). The gate passes when rho >= 0.7 and
  the bootstrap's 2.5% bound is above 0.3.
- noise, per estimator in validation: the population SD of the per-seed means
  (when several encoder seeds were scored) and the bootstrap SD, over deficits,
  of its paired difference to the validation leader (to the runner-up for the
  leader itself; the deficits' own spread is common to all estimators and
  says nothing about a tie). Above 0.02 the tie used for selection must be
  widened, or the two splits pooled.

    python -m ncpfold.validation_check --split receptor \\
        --val lr=val/lr_receptor --val r7=val/r7_receptor \\
        --test lr=outputs_perfold_bcb_p1/scores_receptor --test r7=status/r7_bcb/scores/kch-s0/receptor \\
        --out status/validation_check.csv

Each ``TAG=ROOT`` is an out-root of ``ncpfold.score`` (``replica/**/replica_results.csv``).
An estimator is its classifier/learner cell; an in-context estimator
(CausalPFN, a PFN checkpoint, a panel model) also carries the tag of its root,
so that two checkpoints scored as ``pfn`` stay apart, and the same tag must
name it on both sides. The built-in cells (logistic regression and extra
trees) enter as the single cell that is best in validation (``--builtin
best``), the one a selection would pick, or all of them (``--builtin all``).
Aggregates only are written; rows of other splits already in ``--out`` are kept.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neurocausalpfn.prior import giles_replica as gr

BUILTIN = tuple(getattr(gr, "BUILTIN_CLASSIFIERS", ("logistic_regression", "extra_trees")))
RHO_MIN = 0.7          # the gate's point estimate
RHO_LOW_MIN = 0.3      # and its bootstrap lower bound
NOISE_MAX = 0.02       # validation noise above which the selection tie is widened


def parse_sources(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for it in items:
        tag, sep, root = it.partition("=")
        if not sep or not tag or not root:
            raise ValueError(f"expected TAG=ROOT, got {it!r}")
        out.append((tag, root))
    return out


def load_side(sources: Sequence[Tuple[str, str]], eid: Optional[str] = None,
              label: Optional[str] = None):
    """One side's per-fold rows as ``estimator, seed, deficit, pehe_paper``.

    A built-in cell found under several roots is read from the first one only
    (the same cells are rewritten by every scoring of the representation).
    A root may be named once only (under two tags its in-context estimators
    would count twice), and the side must hold one scoring of the
    representation: one budget, channel and protocol, so that an old scoring
    left in a root (other folds, other test set) is never averaged in."""
    import pandas as pd

    roots = [os.path.normpath(os.path.expanduser(r)) for _, r in sources]
    repeated = sorted({r for r in roots if roots.count(r) > 1})
    if repeated:
        raise SystemExit(f"{len(repeated)} root(s) named under more than one tag; "
                         "give each root a single tag")
    frames, seen_builtin = [], set()
    for tag, root in sources:
        paths = sorted(glob.glob(os.path.join(glob.escape(root), "replica", "**", "replica_results.csv"),
                                 recursive=True))
        if not paths:
            raise SystemExit(f"no replica_results.csv under {root}/replica")
        found = set()
        for p in paths:
            df = pd.read_csv(p)
            if "pehe_paper" not in df.columns:
                continue
            df = df[df["classifier"].fillna("").astype(str) != ""]
            parts = df["representation"].astype(str).str.split("|")
            df = df.assign(eid=parts.str[0], label=parts.str[1],
                           seed=parts.str[2].str.extract(r"seed(\d+)", expand=False).astype(float),
                           scoring=parts.str[3:].str.join("|"))
            if eid is not None:
                df = df[df["eid"] == eid]
            if label is not None:
                df = df[df["label"] == label]
            if df.empty:
                continue
            cell = df["classifier"].astype(str) + "/" + df["learner"].fillna("").astype(str)
            builtin = df["classifier"].isin(BUILTIN)
            df = df.assign(estimator=np.where(builtin, cell, tag + ":" + cell), builtin=builtin)
            keep = ~df["builtin"] | ~df["estimator"].isin(seen_builtin)
            frames.append(df[keep])
            found |= set(df.loc[df["builtin"] & keep, "estimator"])
        seen_builtin |= found
    if not frames:
        raise SystemExit("no rows left after the eid/label filter")
    reps = pd.concat(frames, ignore_index=True)
    if reps[["eid", "label"]].drop_duplicates().shape[0] > 1:
        raise SystemExit("several representations on one side; pass --eid and --label")
    scorings = sorted(reps["scoring"].unique())
    if len(scorings) > 1:
        raise SystemExit("several scorings of the representation on one side (budget|channel|protocol: "
                         + ", ".join(scorings) + "); keep one per root")
    return reps[["estimator", "builtin", "seed", "deficit", "pehe_paper"]]


def _tables(rows):
    """``estimator x deficit`` means (over folds, then seeds) and the per-seed
    means over deficits."""
    per_seed = rows.groupby(["estimator", "seed", "deficit"])["pehe_paper"].mean()
    by_deficit = per_seed.groupby(["estimator", "deficit"]).mean().unstack("deficit")
    seed_means = per_seed.groupby(["estimator", "seed"]).mean()
    return by_deficit, seed_means


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])          # [0]: works before and after scipy 1.9 (.statistic)


def check(val, test, split: str, builtin: str = "best", n_boot: int = 2000,
          seed: int = 0) -> List[Dict]:
    """The ranking row and one row per estimator for one split."""
    val_d, val_seeds = _tables(val)
    test_d, test_seeds = _tables(test)
    if builtin == "best":
        cells = sorted(set(val.loc[val["builtin"], "estimator"]) & set(test_d.index))
        if cells:
            best = val_d.loc[cells].mean(axis=1).idxmin()
            drop = [c for c in cells if c != best]
            val_d, test_d = val_d.drop(drop, errors="ignore"), test_d.drop(drop, errors="ignore")
    elif builtin != "all":
        raise ValueError("builtin must be 'best' or 'all'")
    names = sorted(set(val_d.index) & set(test_d.index))
    deficits = sorted(set(val_d.columns) & set(test_d.columns))
    if len(names) < 3:
        raise SystemExit(f"{split}: {len(names)} estimators on both sides; the ranking needs 3")
    V = val_d.loc[names, deficits].to_numpy(dtype=float)
    T = test_d.loc[names, deficits].to_numpy(dtype=float)
    if not (np.isfinite(V).all() and np.isfinite(T).all()):
        raise SystemExit(f"{split}: some estimator lacks a deficit on one side")
    v_mean, t_mean = V.mean(axis=1), T.mean(axis=1)
    rho = _spearman(v_mean, t_mean)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deficits), size=(n_boot, len(deficits)))
    boot_v = V[:, idx].mean(axis=2)                    # estimators x n_boot
    boot_t = T[:, idx].mean(axis=2)
    rhos = np.array([_spearman(boot_v[:, b], boot_t[:, b]) for b in range(n_boot)])
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = (np.percentile(rhos, [2.5, 97.5]) if len(rhos) else (np.nan, np.nan))

    rank_v = (np.argsort(np.argsort(v_mean)) + 1)
    rank_t = (np.argsort(np.argsort(t_mean)) + 1)
    order = np.argsort(v_mean)
    est_rows, noise = [], []
    for i, nm in enumerate(names):
        sv = val_seeds.loc[nm].to_numpy(dtype=float) if nm in val_seeds.index.get_level_values(0) else np.array([])
        st = test_seeds.loc[nm] if nm in test_seeds.index.get_level_values(0) else []
        seed_sd = float(np.std(sv)) if len(sv) > 1 else float("nan")
        # the deficits' spread is shared by every estimator; a tie is about the
        # paired difference to the validation leader (the runner-up for the leader)
        ref = order[0] if i != order[0] else order[1]
        boot_sd = float(np.std(boot_v[i] - boot_v[ref]))
        noise.append(np.nanmax([seed_sd, boot_sd]))
        est_rows.append({"split": split, "kind": "estimator", "estimator": nm,
                         "val_pehe": float(v_mean[i]), "test_pehe": float(t_mean[i]),
                         "val_rank": int(rank_v[i]), "test_rank": int(rank_t[i]),
                         "val_seed_sd": seed_sd, "val_boot_sd": boot_sd,
                         "n_seeds_val": int(len(sv)), "n_seeds_test": int(len(st)),
                         "noise_over": bool(noise[-1] > NOISE_MAX)})
    max_noise = float(np.nanmax(noise))
    head = {"split": split, "kind": "ranking", "estimator": "",
            "n_estimators": len(names), "n_deficits": len(deficits),
            "rho": rho, "rho_ci_low": float(lo), "rho_ci_high": float(hi), "n_boot": int(len(rhos)),
            "passes": bool(np.isfinite(rho) and rho >= RHO_MIN and lo > RHO_LOW_MIN),
            "max_noise": max_noise, "noise_over": bool(max_noise > NOISE_MAX)}
    return [head] + est_rows


def write(rows: List[Dict], out: str) -> None:
    import pandas as pd

    new = pd.DataFrame(rows)
    if os.path.isfile(out):
        old = pd.read_csv(out)
        new = pd.concat([old[~old["split"].isin(new["split"].unique())], new], ignore_index=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    new.to_csv(out, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="the modality both sides were scored on (row label)")
    ap.add_argument("--val", action="append", required=True, metavar="TAG=ROOT",
                    help="validation out-root (--test-extras scoring); repeat per scoring")
    ap.add_argument("--test", action="append", required=True, metavar="TAG=ROOT",
                    help="test-set out-root (2,828 images); the same tags as --val")
    ap.add_argument("--eid", default="E1")
    ap.add_argument("--label", default="E1")
    ap.add_argument("--builtin", default="best", choices=["best", "all"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    val = load_side(parse_sources(a.val), a.eid, a.label)
    test = load_side(parse_sources(a.test), a.eid, a.label)
    rows = check(val, test, a.split, builtin=a.builtin, n_boot=a.n_boot, seed=a.seed)
    write(rows, a.out)
    h = rows[0]
    print(f"{a.split}: rho {h['rho']:.3f} [{h['rho_ci_low']:.3f}, {h['rho_ci_high']:.3f}] over "
          f"{h['n_estimators']} estimators and {h['n_deficits']} deficits -> "
          f"{'passes' if h['passes'] else 'does not pass'}; max validation noise {h['max_noise']:.3f}"
          f"{' (over 0.02)' if h['noise_over'] else ''}")
    for r in rows[1:]:
        print(f"  {r['estimator']:<40} val {r['val_pehe']:.3f} (#{r['val_rank']})  "
              f"test {r['test_pehe']:.3f} (#{r['test_rank']})  noise {np.nanmax([r['val_seed_sd'], r['val_boot_sd']]):.3f}")


if __name__ == "__main__":
    main()
