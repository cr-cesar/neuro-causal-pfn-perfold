"""Score per-fold latents on the Giles virtual-trial replica (CPU).

    python -m ncpfold.score --data-dir "data/Full data" --atlas-dir data/atlases \\
        --reps outputs_perfold/E5/E5/seed0/folds outputs_perfold/E1/E1/seed0/folds \\
        --out-root outputs_perfold

For every representation folder (10 ``fold*.npz`` files plus ``meta.json``) the
replica's ``evaluate_representation`` is called with a per-fold lookup, so
each test fold is scored with latents produced by an encoder that never saw
it. Results land in ``<out-root>/replica/<eid>/<label>/seed<S>/`` as
``replica_results.csv`` (per deficit, fold, classifier, learner) and
``replica_headline.csv`` (the paper's aggregation; ``pehe_paper_mean`` is the
headline). The anatomical labels of the images are computed once per
(images dir, atlas modality) and cached under ``<out-root>/labels/``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

from neurocausalpfn.prior import giles_replica as gr

from .folds import list_images, modality_dirs, task_dir


def fold_lookup(rep_dir: str, n: int):
    """``(tr_idx, te_idx) -> (Ztr, Zte)`` from the folder's fold*.npz, matched
    to the replica's folds by their test indices (never by call order).
    Same contract as scripts/run_giles_replica.py --fold-latents."""
    by_key: Dict = {}
    for path in sorted(glob.glob(os.path.join(rep_dir, "fold*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            tr, te = z["tr_idx"], z["te_idx"]
            if len(tr) + len(te) != n:
                sys.exit(f"{path}: covers {len(tr) + len(te)} images, expected {n}")
            by_key[(len(te), int(te[0]), int(te[-1]))] = (z["Ztr"].copy(), z["Zte"].copy(),
                                                         tr.copy(), te.copy())
    if not by_key:
        sys.exit(f"no fold*.npz in {rep_dir}")

    def _lookup(tr_idx, te_idx):
        key = (len(te_idx), int(te_idx[0]), int(te_idx[-1]))
        if key not in by_key:
            sys.exit(f"{rep_dir}: no stored fold matches the requested split "
                     f"(different image listing, task or fold count?)")
        Ztr, Zte, tr, te = by_key[key]
        if not (np.array_equal(tr, tr_idx) and np.array_equal(te, te_idx)):
            sys.exit(f"{rep_dir}: stored fold indices differ from the replica's")
        return Ztr, Zte

    _lookup.n_folds = len(by_key)
    return _lookup


def cached_labels(files: Sequence[str], pairs, images_dir: str, modality: str, cache_dir: str):
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha1(("\n".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"labels_{modality}_{os.path.basename(images_dir.rstrip('/'))}_{digest}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    labels = gr.label_images(files, pairs)
    labels.to_csv(path, index=False)
    return labels


def rep_meta(rep_dir: str) -> Dict:
    path = os.path.join(rep_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # fall back to the npz fields
    first = sorted(glob.glob(os.path.join(rep_dir, "fold*.npz")))[0]
    with np.load(first, allow_pickle=False) as z:
        return {"eid": str(z["eid"]), "label": str(z["label"]), "seed": int(z["seed"]),
                "task": str(z["task"]), "n_folds": int(z["n_folds"]), "budget": str(z["budget"])}


def score_reps(rep_dirs: List[str], data_dir: str, atlas_dir: str, out_root: str,
               modality: str = "receptor", scenario_name: str = "ideal",
               deficits=None, with_volume: bool = False, with_nmf: bool = False) -> List[Dict]:
    import pandas as pd

    dirs = modality_dirs(data_dir)
    pairs = gr.load_atlas_pairs(atlas_dir, modality)
    scenario = dict(gr.HEADLINE_SCENARIOS[scenario_name])
    summaries = []
    labels_by_task: Dict[str, object] = {}
    files_by_task: Dict[str, List[str]] = {}

    for rep_dir in rep_dirs:
        meta = rep_meta(rep_dir)
        task = meta.get("task", "disconnectome")
        if task not in files_by_task:
            images_dir = task_dir(dirs, task)
            files_by_task[task] = list_images(images_dir)
            if not files_by_task[task]:
                sys.exit(f"no images under {images_dir}")
            labels_by_task[task] = cached_labels(files_by_task[task], pairs, images_dir,
                                                 modality, os.path.join(out_root, "labels"))
        files, labels = files_by_task[task], labels_by_task[task]
        n_folds = int(meta.get("n_folds", 10))
        lookup = fold_lookup(rep_dir, len(files))
        if lookup.n_folds != n_folds:
            sys.exit(f"{rep_dir}: {lookup.n_folds} folds stored, meta says {n_folds}")

        name = f"{meta['eid']}|{meta['label']}|seed{meta['seed']}|{meta.get('budget', 'chain')}"
        print(f"scoring {name} ({task} task, {modality}, {scenario_name}) ...", flush=True)
        res = gr.evaluate_representation(lookup, labels, pairs, scenario,
                                         n_folds=n_folds, deficits=deficits)
        res.insert(0, "representation", name)
        agg = gr.headline_row(res)
        row = {"representation": name, "eid": meta["eid"], "label": meta["label"],
               "seed": int(meta["seed"]), "budget": meta.get("budget", "chain"),
               "task": task, "modality": modality, **agg, **scenario}

        out_dir = os.path.join(out_root, "replica", meta["eid"],
                               str(meta["label"]).replace("/", "_"), f"seed{meta['seed']}")
        os.makedirs(out_dir, exist_ok=True)
        res.to_csv(os.path.join(out_dir, "replica_results.csv"), index=False)
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, "replica_headline.csv"), index=False)
        print(f"  {name}: PEHE paper {row.get('pehe_paper_mean', float('nan')):.3f}  "
              f"(code-scale {agg['pehe_mean']:.3f}, {agg['classifier']}/{agg['learner']}, "
              f"{agg['n_deficits']} deficits) -> {out_dir}")
        summaries.append(row)

    if with_volume or with_nmf:
        # reference points on the same folds, task = first rep's task
        task = next(iter(files_by_task))
        files, labels = files_by_task[task], labels_by_task[task]
        reps: Dict = {}
        if with_volume:
            v = labels["vol"].to_numpy(dtype=float)
            reps["volume"] = v[:, None] / max(v.max(), 1.0)
        if with_nmf:
            reps["nmf50_perfold"] = _nmf_perfold(files)
        for name, Z in reps.items():
            res = gr.evaluate_representation(Z, labels, pairs, scenario, n_folds=10)
            res.insert(0, "representation", name)
            agg = gr.headline_row(res)
            row = {"representation": name, "eid": "ref", "label": name, "seed": 0,
                   "budget": "-", "task": task, "modality": modality, **agg, **scenario}
            out_dir = os.path.join(out_root, "replica", "ref", name)
            os.makedirs(out_dir, exist_ok=True)
            res.to_csv(os.path.join(out_dir, "replica_results.csv"), index=False)
            pd.DataFrame([row]).to_csv(os.path.join(out_dir, "replica_headline.csv"), index=False)
            print(f"  {name}: PEHE paper {row.get('pehe_paper_mean', float('nan')):.3f}")
            summaries.append(row)
    return summaries


def _nmf_perfold(files):
    """sklearn NMF-50 refitted on each fold's train side (the paper's protocol
    for the reductions; same as run_giles_replica.py --nmf-per-fold)."""
    import nibabel as nib
    from scipy import sparse
    from sklearn.decomposition import NMF

    rows, cols, shape = [], [], None
    for i, path in enumerate(files):
        img = nib.load(path).get_fdata()
        shape = img.shape
        nz = np.flatnonzero(img.ravel() > gr.DISCO_THRESH)
        rows.append(np.full(len(nz), i)); cols.append(nz)
    X = sparse.csr_matrix((np.ones(sum(map(len, cols)), dtype=np.float32),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(files), int(np.prod(shape))))
    k = min(50, len(files) - 1)

    def _refit(tr_idx, te_idx):
        m = NMF(n_components=k, init="nndsvd", max_iter=200, random_state=0, tol=1e-3)
        Ztr = m.fit_transform(X[tr_idx])
        return Ztr.astype(np.float32), m.transform(X[te_idx]).astype(np.float32)

    return _refit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", nargs="+", required=True,
                    help="representation folders (each holding fold*.npz + meta.json)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--atlas-dir", required=True, help="dir containing 2mm_parcellations/")
    ap.add_argument("--out-root", default="outputs_perfold")
    ap.add_argument("--modality", default="receptor", choices=["receptor", "genetics"])
    ap.add_argument("--scenario", default="ideal", choices=sorted(gr.HEADLINE_SCENARIOS))
    ap.add_argument("--deficits", type=int, nargs="*", default=None)
    ap.add_argument("--with-volume", action="store_true", help="also score the volume baseline")
    ap.add_argument("--with-nmf", action="store_true", help="also score per-fold NMF-50 (slow)")
    args = ap.parse_args(argv)
    score_reps(args.reps, args.data_dir, args.atlas_dir, args.out_root, modality=args.modality,
               scenario_name=args.scenario, deficits=args.deficits,
               with_volume=args.with_volume, with_nmf=args.with_nmf)


if __name__ == "__main__":
    main()
