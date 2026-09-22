"""One fold of one variant: build the fold's train folders, train the
encoder(s) on them, encode the whole listing and write the fold latents.

    python -m ncpfold.perfold --eid E5 --run 0 --seed 0 --fold 3 \\
        --data-dir "data/Full data" --out-root outputs_perfold

Output: ``<out-root>/<eid>/<label>/seed<S>/folds/fold<K>.npz`` with Ztr, Zte,
tr_idx, te_idx and files, the exact layout ``scripts/run_giles_replica.py
--fold-latents`` consumes, plus ``meta.json`` next to it. The checkpoints and
logs of the training go to ``<out-root>/<eid>/<label>/seed<S>/fold<K>/``.

The training seed is ``1000 * seed + fold`` so folds of one study seed start
from different initialisations while staying reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .catalogue import find_run
from .encoders import train_and_encode
from .folds import (N_FOLDS, list_images, make_fold_dirs, modality_dirs,
                    task_dir)


def run_fold(eid: str, run, label, seed: int, fold: int, data_dir: str, out_root: str,
             task: str = "disconnectome", budget: str = "chain", n_folds: int = N_FOLDS,
             overrides=None, device: str = "auto", force: bool = False) -> str:
    spec = find_run(eid, run=run, label=label)
    dirs = modality_dirs(data_dir)
    canonical_paths = list_images(task_dir(dirs, task))
    if not canonical_paths:
        sys.exit(f"no images under {task_dir(dirs, task)}")

    var_dir = os.path.join(out_root, eid, spec.label.replace("/", "_"), f"seed{seed}")
    folds_dir = os.path.join(var_dir, "folds")
    npz_path = os.path.join(folds_dir, f"fold{fold}.npz")
    if os.path.exists(npz_path) and not force:
        print(f"exists, skipping: {npz_path}")
        return npz_path

    fold_info = make_fold_dirs(os.path.join(out_root, "folds"), dirs, canonical_paths, fold, n_folds)
    tr_idx, te_idx, names = fold_info["tr_idx"], fold_info["te_idx"], fold_info["files"]
    print(f"{eid} {spec.label} seed {seed} fold {fold}/{n_folds}: "
          f"{len(tr_idx)} train / {len(te_idx)} test | kind {spec.kind} | budget {budget}")

    t0 = time.time()
    Z = train_and_encode(spec, 1000 * seed + fold, fold_info["dirs"], dirs, names,
                         os.path.join(var_dir, f"fold{fold}"), budget=budget,
                         overrides=overrides, device=device)
    os.makedirs(folds_dir, exist_ok=True)
    np.savez(npz_path, Ztr=Z[tr_idx], Zte=Z[te_idx], tr_idx=tr_idx, te_idx=te_idx,
             files=np.array(names), fold=fold, n_folds=n_folds, dim=Z.shape[1],
             eid=np.array(eid), label=np.array(spec.label), seed=seed,
             budget=np.array(budget), task=np.array(task))
    meta = {"eid": eid, "label": spec.label, "kind": spec.kind, "seed": seed,
            "budget": budget, "task": task, "n_folds": n_folds, "dim": int(Z.shape[1]),
            "meta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.meta.items()}}
    with open(os.path.join(folds_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {npz_path}  Ztr{Z[tr_idx].shape} Zte{Z[te_idx].shape}  "
          f"({(time.time() - t0) / 3600:.2f} h)")
    return npz_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eid", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", type=int, default=None, help="variant index within the eid (see plan)")
    g.add_argument("--label", default=None, help="exact variant label, e.g. 'E3[backbone=cnn]'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, required=True, help="0-based fold index")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--data-dir", required=True, help='tier folder, e.g. "data/Full data"')
    ap.add_argument("--out-root", default="outputs_perfold")
    ap.add_argument("--task", default="disconnectome", choices=["disconnectome", "lesion"],
                    help="which modality's listing defines the folds (the replica task)")
    ap.add_argument("--budget", default="chain", choices=["chain", "published"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overrides", default=None, help="JSON with smoke knobs (tests only)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    overrides = json.loads(args.overrides) if args.overrides else None
    run_fold(args.eid, args.run, args.label, args.seed, args.fold, args.data_dir,
             args.out_root, task=args.task, budget=args.budget, n_folds=args.n_folds,
             overrides=overrides, device=args.device, force=args.force)


if __name__ == "__main__":
    main()
