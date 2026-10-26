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
headline). A variant's ``folds/`` folder is its primary channel (the
disconnectome export for two-encoder variants); ``folds_lesion`` and
``folds_both`` score the other channels when wanted. The anatomical labels of the images are computed once per
(images dir, atlas modality) and cached under ``<out-root>/labels/``.

``--ensemble`` concatenates, fold by fold, the latents of every ``--reps`` folder
that shares eid, label, budget and channel across seeds (a seed ensemble: 3
seeds of a 50-d encoder give one 150-d representation) and scores it as
``<label>+x<n>``; ``--ensemble-only`` skips the single seeds. Each fold still
only uses encoders that never saw its test images.

``--test-singles`` keeps the trained folds but scores only the patients with a
single image (the paper's evaluation set). In 'giles' group mode the later
acquisitions of a multi-image patient train every fold's encoder, so that
patient's tested image is not independent of its encoder; single-image
patients are, whatever the mode. Protocol and output folder get a
``-singles`` suffix.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from neurocausalpfn.prior import giles_replica as gr

from .folds import all_folds, list_images, load_groups, modality_dirs, task_dir


# the fold whose latents the lookup served last: the replica calls the lookup
# once per fold before scoring it, so a per-fold estimator reads its fold here
CURRENT: Dict[str, int] = {"fold": -1}


class PerFoldEstimator:
    """One trained Neuro-Causal-PFN per fold (``{fold}`` in the checkpoint
    path): PFN k, pre-trained on a pool that excludes fold k's test patients,
    scores fold k only. Checkpoints are loaded on first use."""

    def __init__(self, pattern: str, **kw):
        if "{fold}" not in pattern:
            raise ValueError("a per-fold estimator needs '{fold}' in its checkpoint path")
        self.pattern, self.kw, self.loaded = pattern, kw, {}
        self.name = "pfn"

    def path(self, fold: int) -> str:
        return self.pattern.replace("{fold}", str(fold))

    def __call__(self, Xtr, W, Y, Xte):
        from neurocausalpfn.pfn.estimator import PFNEstimator

        fold = CURRENT["fold"]
        if fold < 0:
            raise RuntimeError("per-fold estimator called before any fold lookup")
        if fold not in self.loaded:
            path = self.path(fold)
            if not os.path.exists(path):
                sys.exit(f"no checkpoint for fold {fold}: {path}")
            self.loaded = {fold: PFNEstimator(path, **self.kw)}     # one checkpoint in memory
        return self.loaded[fold](Xtr, W, Y, Xte)


def fold_lookup(rep_dir: str, n: int):
    """``(tr_idx, te_idx) -> (Ztr, Zte)`` from the folder's fold*.npz, matched
    to the replica's folds by their test indices (never by call order).
    Same contract as scripts/run_giles_replica.py --fold-latents."""
    by_key: Dict = {}
    fold_of: Dict = {}
    # glob.escape: variant labels carry brackets (E3[backbone=cnn]) that glob
    # would otherwise read as a character class
    for k, path in enumerate(sorted(glob.glob(os.path.join(glob.escape(rep_dir), "fold*.npz")))):
        with np.load(path, allow_pickle=False) as z:
            tr, te = z["tr_idx"], z["te_idx"]
            if len(tr) + len(te) != n:
                sys.exit(f"{path}: covers {len(tr) + len(te)} images, expected {n}")
            key = (len(te), int(te[0]), int(te[-1]))
            by_key[key] = (z["Ztr"].copy(), z["Zte"].copy(), tr.copy(), te.copy())
            fold_of[key] = int(z["fold"]) if "fold" in z.files else k
    if not by_key:
        sys.exit(f"no fold*.npz in {rep_dir}")

    owner = {int(i): key for key, (_, _, _, te) in by_key.items() for i in te}

    def _lookup(tr_idx, te_idx):
        key = (len(te_idx), int(te_idx[0]), int(te_idx[-1]))
        if key in by_key:
            Ztr, Zte, tr, te = by_key[key]
            if not (np.array_equal(tr, tr_idx) and np.array_equal(te, te_idx)):
                sys.exit(f"{rep_dir}: stored fold indices differ from the replica's")
            CURRENT["fold"] = fold_of[key]
            return Ztr, Zte
        # a subset of one stored test fold (e.g. --test-singles): same training
        # side, rows of Zte picked by position
        key = owner.get(int(te_idx[0]))
        if key is None:
            # validation extras (--test-extras): both sides lie inside one stored
            # fold's TRAIN side (the never-tested images are train-only in every
            # fold); the stored fold is the one whose train side holds every
            # requested image, which is unique because the other folds' train
            # sides lack the images they test
            want = set(int(i) for i in tr_idx) | set(int(i) for i in te_idx)
            for k2, (Ztr, Zte, tr, te) in by_key.items():
                trs = set(int(i) for i in tr)
                if want <= trs:
                    pos = {int(i): p for p, i in enumerate(tr)}
                    CURRENT["fold"] = fold_of[k2]
                    return Ztr[[pos[int(i)] for i in tr_idx]], Ztr[[pos[int(i)] for i in te_idx]]
            sys.exit(f"{rep_dir}: no stored fold matches the requested split "
                     f"(different image listing, task or fold count?)")
        Ztr, Zte, tr, te = by_key[key]
        pos = {int(i): p for p, i in enumerate(te)}
        if not np.array_equal(tr, tr_idx) or any(int(i) not in pos for i in te_idx):
            sys.exit(f"{rep_dir}: requested test images are not a subset of one stored fold")
        CURRENT["fold"] = fold_of[key]
        return Ztr, Zte[[pos[int(i)] for i in te_idx]]

    _lookup.n_folds = len(by_key)
    return _lookup


def ensemble_lookup(lookups):
    """Column-wise concatenation of several per-fold lookups (one per seed):
    the seed ensemble of a representation, fold by fold."""
    def _lookup(tr_idx, te_idx):
        parts = [lk(tr_idx, te_idx) for lk in lookups]
        return (np.concatenate([a for a, _ in parts], axis=1),
                np.concatenate([b for _, b in parts], axis=1))
    _lookup.n_folds = lookups[0].n_folds
    return _lookup


def ensemble_units(rep_dirs: Sequence[str], ensemble: bool, ensemble_only: bool):
    """``[(meta, [rep_dir, ...])]`` to score: every folder on its own and, with
    ``ensemble``, one unit per group of folders sharing eid/label/budget/
    channel/task across seeds (label ``<label>+x<n>``, seed 0)."""
    units = [(rep_meta(d), [d]) for d in rep_dirs]
    if not ensemble:
        return units
    groups: Dict[tuple, list] = {}
    for meta, dirs in units:
        key = (meta["eid"], str(meta["label"]), meta.get("budget", "chain"),
               meta.get("channel", "primary"), meta.get("task", "disconnectome"))
        groups.setdefault(key, []).append((int(meta["seed"]), meta, dirs[0]))
    out = [] if ensemble_only else list(units)
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort()
        meta = dict(members[0][1]); meta["label"] = f"{meta['label']}+x{len(members)}"
        meta["seed"] = 0; meta["seeds"] = [m[0] for m in members]
        out.append((meta, [m[2] for m in members]))
    return out


def singles_mask(names: Sequence[str], groups: Optional[Dict[str, tuple]]) -> np.ndarray:
    """True for images whose group holds a single image in this listing (the
    paper's evaluation set: patients with one acquisition). Without a group
    table every image counts as a single."""
    if groups is None:
        return np.ones(len(names), dtype=bool)
    gid = [groups.get(nm, (f"__solo__{nm}", 0))[0] for nm in names]
    counts: Dict[str, int] = {}
    for g in gid:
        counts[g] = counts.get(g, 0) + 1
    return np.array([counts[g] == 1 for g in gid], dtype=bool)


def restrict_test(folds, mask: np.ndarray):
    """Same training sides; test sides reduced to the masked images."""
    return [(tr, te[mask[te]]) for tr, te in folds]


def extras_mask(names: Sequence[str], groups) -> np.ndarray:
    """True for the images that are never tested under the group-aware
    protocol: a later acquisition (rank > 0), or a member of a group without
    a rank-0 image; with an exported fold table, the rows whose fold is -1.
    They train in every fold and are the validation set of the Phase-2
    selection (the paper's rule: nothing is chosen on the tested patients)."""
    if groups is None:
        raise ValueError("the validation extras need the group table")
    fold_of = getattr(groups, "folds", None)
    if fold_of is not None:
        return np.array([fold_of.get(nm, 0) < 0 for nm in names], dtype=bool)
    has_rank0: Dict[str, bool] = {}
    for nm in names:
        g, rank = groups.get(nm, (f"__solo__{nm}", 0))
        has_rank0[g] = has_rank0.get(g, False) or rank == 0
    out = []
    for nm in names:
        g, rank = groups.get(nm, (f"__solo__{nm}", 0))
        out.append(rank > 0 or not has_rank0[g])
    return np.array(out, dtype=bool)


def extras_folds(folds, mask: np.ndarray, names: Sequence[str]):
    """Validation folds on the never-tested images: pseudo-fold k tests the
    extras whose name hashes to k (fixed, independent of the listing order)
    and trains on stored fold k's training side without them. Every extra is
    tested exactly once; no tested image of the benchmark is ever a query."""
    import hashlib

    n_folds = len(folds)
    bucket = {i: int(hashlib.sha1(str(names[i]).encode()).hexdigest(), 16) % n_folds
              for i in np.flatnonzero(mask)}
    out = []
    for k, (tr, te) in enumerate(folds):
        te_k = np.array(sorted(i for i, b in bucket.items() if b == k), dtype=int)
        missing = [i for i in te_k if i not in set(int(j) for j in tr)]
        if missing:
            raise ValueError(f"{len(missing)} validation images are not in fold {k}'s training side")
        tr_k = np.array(sorted(set(int(j) for j in tr) - set(int(i) for i in te_k)), dtype=int)
        out.append((tr_k, te_k))
    return out


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
    first = sorted(glob.glob(os.path.join(glob.escape(rep_dir), "fold*.npz")))[0]
    with np.load(first, allow_pickle=False) as z:
        return {"eid": str(z["eid"]), "label": str(z["label"]), "seed": int(z["seed"]),
                "task": str(z["task"]), "n_folds": int(z["n_folds"]), "budget": str(z["budget"])}


def resolve_classifiers(classifiers: Optional[Sequence[str]], estimators: Sequence[str] = (),
                        only_estimators: bool = False, warm_up: bool = True) -> List[str]:
    """The replica's classifier list for a scoring: ``classifiers`` (default
    logistic regression and extra trees; a name outside the built-ins is a
    member of the tabular foundation-model panel, ``tabpfn`` / ``tabicl``,
    registered through ``neurocausalpfn.pfn.panel`` and warmed up now so a
    missing weight file fails before the scoring) followed by the in-context
    ``estimators``; ``only_estimators`` keeps the estimators alone."""
    base = list(classifiers) if classifiers else list(gr.BUILTIN_CLASSIFIERS)
    panel_names = [c for c in base if c not in gr.BUILTIN_CLASSIFIERS]
    if panel_names:
        from neurocausalpfn.pfn.panel import register_panel
        register_panel(panel_names, warm_up=warm_up)
    names = []
    for name in estimators:
        if name == "causalpfn":
            from neurocausalpfn.pfn.estimator import CausalPFNEstimator
            gr.register_estimator("causalpfn", CausalPFNEstimator())
        elif name.endswith(".pt") and "{fold}" in name:
            gr.register_estimator("pfn", PerFoldEstimator(name)); name = "pfn"
        elif name.endswith(".pt"):
            from neurocausalpfn.pfn.estimator import PFNEstimator
            gr.register_estimator("pfn", PFNEstimator(name)); name = "pfn"
        else:
            raise ValueError(f"unknown estimator {name!r}")
        names.append(name)
    if only_estimators and names:
        return names                          # the design's Tier 4: the fixed estimator alone
    return base + names


def score_reps(rep_dirs: List[str], data_dir: str, atlas_dir: str, out_root: str,
               modality: str = "receptor", scenario_name: str = "ideal",
               deficits=None, with_volume: bool = False, with_nmf: bool = False,
               groups_csv: Optional[str] = None, test_singles: bool = False,
               test_extras: bool = False,
               estimators: Sequence[str] = (), only_estimators: bool = False,
               ensemble: bool = False, ensemble_only: bool = False,
               only_folds: Optional[Sequence[int]] = None,
               classifiers: Optional[Sequence[str]] = None) -> List[Dict]:
    """``estimators``: extra in-context estimators registered in the main
    package's replica (``causalpfn`` = the off-the-shelf CausalPFN with fixed
    weights, the design's Tier-4 evaluator; a checkpoint path, or one with
    ``{fold}`` for one checkpoint per fold); they are scored next to the
    logistic regression and extra trees on the same folds. ``classifiers``
    replaces that pair (``tabpfn``, ``tabicl``: the foundation-model panel,
    scored one-model and two-model like the pair). ``only_folds`` scores
    those folds alone (held-out folds of a pool experiment); the protocol and
    the output folder carry the fold list."""
    import pandas as pd

    classifiers = resolve_classifiers(classifiers, estimators, only_estimators)
    print("classifiers:", " ".join(classifiers), flush=True)

    groups = load_groups(groups_csv)
    dirs = modality_dirs(data_dir)
    pairs = gr.load_atlas_pairs(atlas_dir, modality)
    scenario = dict(gr.HEADLINE_SCENARIOS[scenario_name])
    summaries = []
    labels_by_task: Dict[str, object] = {}
    files_by_task: Dict[str, List[str]] = {}

    for meta, unit_dirs in ensemble_units(rep_dirs, ensemble, ensemble_only):
        rep_dir = unit_dirs[0]
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
        lookup = fold_lookup(rep_dir, len(files)) if len(unit_dirs) == 1 else \
            ensemble_lookup([fold_lookup(d, len(files)) for d in unit_dirs])
        if lookup.n_folds != n_folds:
            sys.exit(f"{rep_dir}: {lookup.n_folds} folds stored, meta says {n_folds}")
        gmode = meta.get("group_mode") or None
        if gmode and groups is None:
            sys.exit(f"{rep_dir} was trained on group-aware folds ({gmode}); pass --groups")
        if not gmode and groups is not None:
            sys.exit(f"{rep_dir} was trained on image-level folds; drop --groups")
        names = [os.path.basename(f) for f in files]
        folds = all_folds(len(files), n_folds, names=names, groups=groups, mode=gmode or "giles")
        if test_singles and test_extras:
            sys.exit("--test-singles and --test-extras are different test sets; pass one")
        if test_singles:
            # evaluate only single-acquisition patients: none of them was in the
            # training side of its fold's encoder, whatever the group mode
            folds = restrict_test(folds, singles_mask(names, groups))
        if test_extras:
            # the validation set: the never-tested images, in pseudo-folds
            folds = extras_folds(folds, extras_mask(names, groups), names)
        fold_tag = ""
        if only_folds is not None:
            keep = sorted(set(int(k) for k in only_folds))
            if any(k < 0 or k >= len(folds) for k in keep):
                sys.exit(f"--only-folds {keep}: the representation has {len(folds)} folds")
            folds = [folds[k] for k in keep]
            fold_tag = "-f" + ",".join(str(k) for k in keep)

        channel = meta.get("channel", "primary")
        suffix = "-singles" if test_singles else ("-extras" if test_extras else "")
        protocol = (f"grp-{gmode}" if gmode else "img") + suffix + fold_tag
        name = f"{meta['eid']}|{meta['label']}|seed{meta['seed']}|{meta.get('budget', 'chain')}|{channel}|{protocol}"
        print(f"scoring {name} ({task} task, {modality}, {scenario_name}) ...", flush=True)
        res = gr.evaluate_representation(lookup, labels, pairs, scenario,
                                         n_folds=n_folds, deficits=deficits, folds=folds,
                                         classifiers=classifiers)
        res.insert(0, "representation", name)
        agg = gr.headline_row(res)
        row = {"representation": name, "eid": meta["eid"], "label": meta["label"],
               "seed": int(meta["seed"]), "budget": meta.get("budget", "chain"),
               "channel": channel, "protocol": protocol, "task": task, "modality": modality,
               **agg, **scenario}

        out_dir = os.path.join(out_root, "replica", meta["eid"],
                               str(meta["label"]).replace("/", "_"), f"seed{meta['seed']}",
                               channel + suffix + fold_tag)
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
        ref_names = [os.path.basename(f) for f in files]
        ref_folds = all_folds(len(files), 10, names=ref_names,
                              groups=groups, mode="giles") if groups is not None else None
        ref_protocol = "grp-giles" if groups is not None else "img"
        if test_singles:
            ref_folds = restrict_test(ref_folds if ref_folds is not None
                                      else all_folds(len(files), 10), singles_mask(ref_names, groups))
            ref_protocol += "-singles"
        reps: Dict = {}
        if with_volume:
            v = labels["vol"].to_numpy(dtype=float)
            reps["volume"] = v[:, None] / max(v.max(), 1.0)
        if with_nmf:
            reps["nmf50_perfold"] = _nmf_perfold(files)
        for name, Z in reps.items():
            res = gr.evaluate_representation(Z, labels, pairs, scenario, n_folds=10, folds=ref_folds)
            res.insert(0, "representation", name)
            agg = gr.headline_row(res)
            row = {"representation": name, "eid": "ref", "label": name, "seed": 0,
                   "budget": "-", "channel": "-", "protocol": ref_protocol, "task": task,
                   "modality": modality, **agg, **scenario}
            out_dir = os.path.join(out_root, "replica", "ref", name + ("-singles" if test_singles else ""))
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
    ap.add_argument("--groups", default=None,
                    help="the same filename,group,rank CSV the folds were trained with")
    ap.add_argument("--estimators", nargs="*", default=[],
                    help="extra in-context estimators: 'causalpfn' (off-the-shelf, fixed weights), "
                         "a trained Neuro-Causal-PFN checkpoint (pfn.pt), or a path with '{fold}' "
                         "for one checkpoint per fold")
    ap.add_argument("--only-estimators", action="store_true",
                    help="score the --estimators alone (no logistic regression / extra trees)")
    ap.add_argument("--classifiers", nargs="+", default=None,
                    help="classifiers fitted per slice, one-model and two-model (default "
                         "logistic_regression extra_trees); 'tabpfn' and 'tabicl' are the "
                         "tabular foundation-model panel (separate venv, GPU)")
    ap.add_argument("--ensemble", action="store_true",
                    help="also score the seed ensemble of every group of --reps sharing "
                         "eid/label/budget/channel (latents concatenated fold by fold)")
    ap.add_argument("--ensemble-only", action="store_true",
                    help="with --ensemble: skip the single seeds")
    ap.add_argument("--only-folds", type=int, nargs="+", default=None,
                    help="score these folds only (e.g. the held-out folds of a pool experiment)")
    ap.add_argument("--test-extras", action="store_true",
                    help="validation set: the never-tested images (later acquisitions and groups "
                         "without a rank-0 image), in pseudo-folds, trained on the stored fold's "
                         "training side without them; needs --groups")
    ap.add_argument("--test-singles", action="store_true",
                    help="evaluate only patients with a single image (the paper's evaluation "
                         "set); training sides and stored folds unchanged, so no image of a "
                         "tested patient was ever in its fold's encoder")
    args = ap.parse_args(argv)
    score_reps(args.reps, args.data_dir, args.atlas_dir, args.out_root, modality=args.modality,
               scenario_name=args.scenario, deficits=args.deficits,
               with_volume=args.with_volume, with_nmf=args.with_nmf, groups_csv=args.groups,
               test_singles=args.test_singles, test_extras=args.test_extras, estimators=args.estimators,
               only_estimators=args.only_estimators, ensemble=args.ensemble,
               ensemble_only=args.ensemble_only, only_folds=args.only_folds,
               classifiers=args.classifiers)


if __name__ == "__main__":
    main()
