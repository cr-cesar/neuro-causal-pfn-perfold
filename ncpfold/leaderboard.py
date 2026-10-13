"""Collect every replica headline of the study into one table.

    python -m ncpfold.leaderboard --out-root outputs_perfold

Reads ``<out-root>/replica/**/replica_headline.csv`` and aggregates the
paper-scale PEHE over seeds per (eid, label, budget, task, channel): mean, population
std and n. Writes ``leaderboard.csv`` and ``leaderboard.md`` under out-root.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np


def _eid_key(eid: str):
    m = re.match(r"^E(\d+)([a-z]?)$", str(eid))
    return (int(m.group(1)), m.group(2)) if m else (999, str(eid))


def build(out_root: str):
    import pandas as pd

    paths = glob.glob(os.path.join(out_root, "replica", "**", "replica_headline.csv"), recursive=True)
    if not paths:
        raise SystemExit(f"no replica_headline.csv under {out_root}/replica")
    frames = []
    for p in paths:
        h = pd.read_csv(p)
        # headlines written before the CI columns existed (or with an empty CI)
        # are recomputed from the per-deficit results next to them, so that a
        # leaderboard regenerated today carries the deficit CI for every cell
        if "pehe_paper_ci_low" not in h.columns or not np.isfinite(h["pehe_paper_ci_low"]).all():
            results = os.path.join(os.path.dirname(p), "replica_results.csv")
            if os.path.isfile(results):
                from neurocausalpfn.prior import giles_replica as gr
                agg = gr.headline_row(pd.read_csv(results))
                for k in ("pehe_paper_ci_low", "pehe_paper_ci_high", "pehe_paper_selected",
                          "pehe_paper_selected_ci_low", "pehe_paper_selected_ci_high"):
                    if k in agg:
                        h[k] = agg[k]
        frames.append(h)
    df = pd.concat(frames, ignore_index=True)
    if "protocol" not in df.columns:
        df["protocol"] = "img"
    df["protocol"] = df["protocol"].fillna("img")
    keys = ["eid", "label", "budget", "task", "channel", "protocol"]
    rows = []
    for key, sub in df.groupby(keys, dropna=False):
        v = sub["pehe_paper_mean"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        def col(name):
            return float(sub[name].mean()) if name in sub and np.isfinite(sub[name]).any() else float("nan")
        rows.append({**dict(zip(keys, key)), "n_seeds": int(len(v)),
                     "pehe_paper_mean": float(np.mean(v)) if len(v) else float("nan"),
                     # population SD of the seed means (folds and simulated outcomes are
                     # shared by the seeds): the encoder's seed noise, not a CI
                     "pehe_paper_seed_sd": float(np.std(v)) if len(v) else float("nan"),
                     # Giles's interval: 95% CI over the 16 deficits, mean over seeds
                     "pehe_paper_ci_low": col("pehe_paper_ci_low"),
                     "pehe_paper_ci_high": col("pehe_paper_ci_high"),
                     # the paper-scale value of the cell the paper scale would select
                     # (the headline cell is selected on the released-code scale)
                     "pehe_paper_selected": col("pehe_paper_selected"),
                     "balacc_mean": col("balacc_mean"),
                     "pehe_code_mean": col("pehe_mean")})
    board = pd.DataFrame(rows)
    board["_k"] = board["eid"].map(_eid_key)
    board = board.sort_values(["_k", "pehe_paper_mean"]).drop(columns="_k")
    board.to_csv(os.path.join(out_root, "leaderboard.csv"), index=False)

    lines = ["# Per-fold study leaderboard", "",
             "PEHE on the virtual-trial replica, paper definition, every representation "
             "fitted inside each fold (no anatomical leakage). Mean over seeds; "
             "seed SD = population SD of the seed means (not a CI); 95% CI = over the 16 "
             "deficits (Giles's interval). Lower is better.", "",
             "| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper (seed SD) | 95% CI over deficits | Bal. acc. |",
             "|-----|---------|--------|------|---------|-------|-------|----------------------|----------------------|-----------|"]
    for r in board.itertuples():
        ci = f"{r.pehe_paper_ci_low:.3f}–{r.pehe_paper_ci_high:.3f}" if np.isfinite(r.pehe_paper_ci_low) else "–"
        lines.append(f"| {r.eid} | {r.label} | {r.budget} | {r.task} | {r.channel} | {r.protocol} | {r.n_seeds} | "
                     f"{r.pehe_paper_mean:.3f} ({r.pehe_paper_seed_sd:.3f}) | {ci} | {r.balacc_mean:.3f} |")
    with open(os.path.join(out_root, "leaderboard.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return board


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="outputs_perfold")
    args = ap.parse_args(argv)
    board = build(args.out_root)
    print(board.to_string(index=False))
    print(f"\nwritten: {args.out_root}/leaderboard.csv, leaderboard.md")


if __name__ == "__main__":
    main()
