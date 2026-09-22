"""Fold definition and per-fold training directories.

The folds are exactly the replica's: ``KFold(10, shuffle=True, random_state=0)``
over the SORTED file listing of the task modality's directory (the same call
``scripts/run_giles_replica.py`` and ``scripts/train_giles_style_vae50.py`` make
in the main repository). The split is therefore fixed before any model is
fitted, as the paper's Methods require.

A fold's training directory is a folder of symlinks to the train-side files of
each modality. Every trainer of the main package takes its data root from its
config, so pointing that root at the symlink folder restricts training to the
fold's train side without touching the trainers.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

N_FOLDS = 10
FOLD_SEED = 0          # the replica's KFold random_state
MODALITIES = ("lesions", "disconnectomes")


def list_images(directory: str) -> List[str]:
    """The replica's listing: sorted ``*.nii*`` paths of one directory."""
    return sorted(glob.glob(os.path.join(directory, "*.nii*")))


def basenames(paths: Sequence[str]) -> List[str]:
    return [os.path.basename(p) for p in paths]


def all_folds(n: int, n_folds: int = N_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    return [(tr, te) for tr, te in
            KFold(n_splits=n_folds, shuffle=True, random_state=FOLD_SEED).split(np.arange(n))]


def fold_indices(n: int, fold: int, n_folds: int = N_FOLDS) -> Tuple[np.ndarray, np.ndarray]:
    folds = all_folds(n, n_folds)
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} out of range for {n_folds} folds")
    return folds[fold]


def modality_dirs(data_dir: str) -> Dict[str, str]:
    """``{"lesions": <dir>, "disconnectomes": <dir>}`` under a tier folder such
    as ``data/Full data``."""
    out = {}
    for kind in MODALITIES:
        d = os.path.join(data_dir, kind)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing modality folder {d}")
        out[kind] = d
    return out


def task_dir(dirs: Dict[str, str], task: str) -> str:
    """The replica's ``--images-dir`` for a task ('disconnectome' or 'lesion')."""
    return dirs["disconnectomes" if task.startswith("disc") else "lesions"]


def _relink(dst: str, src_by_name: Dict[str, str], names: Sequence[str]) -> None:
    """Make ``dst`` hold exactly the symlinks ``names`` -> sources. Tolerant to
    concurrent callers (several array jobs build the same fold at once): a
    link another process just created or removed is not an error."""
    os.makedirs(dst, exist_ok=True)
    wanted = set(names)
    for existing in os.listdir(dst):
        if existing not in wanted:
            try:
                os.remove(os.path.join(dst, existing))
            except FileNotFoundError:
                pass
    for name in names:
        link = os.path.join(dst, name)
        target = os.path.abspath(src_by_name[name])
        for _attempt in range(3):
            try:
                if os.path.islink(link):
                    if os.readlink(link) == target:
                        break
                    os.remove(link)
                elif os.path.exists(link):
                    os.remove(link)
                os.symlink(target, link)
                break
            except FileExistsError:
                continue                      # created by a concurrent job; re-check
            except FileNotFoundError:
                continue                      # removed by a concurrent job; re-create
        else:
            raise RuntimeError(f"could not create symlink {link}")


def make_fold_dirs(fold_root: str, dirs: Dict[str, str], canonical: Sequence[str],
                   fold: int, n_folds: int = N_FOLDS) -> Dict[str, object]:
    """Create (idempotently) ``fold_root/fold{K}/{lesions,disconnectomes}`` with
    symlinks to the train-side files of fold K, for both modalities.

    ``canonical`` is the task modality's sorted listing; the train side is
    chosen by basename, so both modality folders hold the same ids. A file of
    the canonical listing missing from the other modality is an error: the
    paired trainers would silently drop it.
    """
    names = basenames(canonical)
    if len(set(names)) != len(names):
        raise ValueError("duplicate basenames in the canonical listing")
    tr_idx, te_idx = fold_indices(len(names), fold, n_folds)
    train_names = [names[i] for i in tr_idx]

    made = {}
    for kind, src in dirs.items():
        src_by_name = {os.path.basename(p): p for p in list_images(src)}
        missing = [n for n in train_names if n not in src_by_name]
        if missing:
            raise FileNotFoundError(f"{len(missing)} canonical files missing from {src}, "
                                    f"e.g. {missing[:3]}")
        dst = os.path.join(fold_root, f"fold{fold}", kind)
        _relink(dst, src_by_name, train_names)
        made[kind] = dst

    manifest = {"fold": fold, "n_folds": n_folds, "n": len(names),
                "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                "te_idx": [int(i) for i in te_idx], "kfold_seed": FOLD_SEED}
    with open(os.path.join(fold_root, f"fold{fold}", "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return {"dirs": made, "tr_idx": tr_idx, "te_idx": te_idx, "files": names}


def align_to_canonical(Z: np.ndarray, names: Sequence[str], canonical_names: Sequence[str]) -> np.ndarray:
    """Reorder rows of ``Z`` (one per ``names``) into the canonical order. Every
    canonical file must be present exactly once."""
    index = {n: i for i, n in enumerate(canonical_names)}
    out = np.full((len(canonical_names), Z.shape[1]), np.nan, dtype=np.float32)
    seen = np.zeros(len(canonical_names), dtype=bool)
    for z, n in zip(Z, names):
        i = index.get(n)
        if i is None:
            raise KeyError(f"encoded file {n} is not in the canonical listing")
        out[i] = z
        seen[i] = True
    if not seen.all():
        missing = [canonical_names[i] for i in np.flatnonzero(~seen)[:3]]
        raise ValueError(f"{int((~seen).sum())} canonical files were not encoded, e.g. {missing}")
    return out
