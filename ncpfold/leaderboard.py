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
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    keys = ["eid", "label", "budget", "task", "channel"]
    rows = []
    for key, sub in df.groupby(keys, dropna=False):
        v = sub["pehe_paper_mean"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({**dict(zip(keys, key)), "n_seeds": int(len(v)),
                     "pehe_paper_mean": float(np.mean(v)) if len(v) else float("nan"),
                     "pehe_paper_std": float(np.std(v)) if len(v) else float("nan"),
                     "balacc_mean": float(sub["balacc_mean"].mean()) if "balacc_mean" in sub else float("nan"),
                     "pehe_code_mean": float(sub["pehe_mean"].mean())})
    board = pd.DataFrame(rows)
    board["_k"] = board["eid"].map(_eid_key)
    board = board.sort_values(["_k", "pehe_paper_mean"]).drop(columns="_k")
    board.to_csv(os.path.join(out_root, "leaderboard.csv"), index=False)

    lines = ["# Per-fold study leaderboard", "",
             "PEHE on the virtual-trial replica, paper definition, every representation "
             "fitted inside each fold (no anatomical leakage). Mean over seeds "
             "(population std). Lower is better.", "",
             "| Exp | Variant | Budget | Task | Channel | Seeds | PEHE paper | Bal. acc. |",
             "|-----|---------|--------|------|---------|-------|------------|-----------|"]
    for r in board.itertuples():
        lines.append(f"| {r.eid} | {r.label} | {r.budget} | {r.task} | {r.channel} | {r.n_seeds} | "
                     f"{r.pehe_paper_mean:.3f} ({r.pehe_paper_std:.3f}) | {r.balacc_mean:.3f} |")
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
