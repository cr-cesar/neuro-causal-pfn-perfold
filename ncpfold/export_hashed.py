"""Export a cohort for training on a machine outside the cluster, with every
file name replaced by a keyed hash and the fold membership fixed by name.

The per-fold protocol defines its folds over the SORTED file listing, so a
cohort copied under other names would get other folds. This export computes
the folds here, on the real listing and the real group table, and writes a
fold table that pins them::

    HASH_SALT_FILE=~/.cache_salt python -m ncpfold.export_hashed \\
        --data-dir ~/Scratch/data_bcb/Full\\ data --groups outputs_perfold/groups_public.csv \\
        --dst ~/Scratch/export_bcb

Result (nothing identifiable: no name, no clinical field, headers scrubbed)::

    <dst>/Full data/lesions/<hash>.nii.gz          every image of the listing, both modalities
    <dst>/Full data/disconnectomes/<hash>.nii.gz
    <dst>/fold_table.csv                            filename,group,rank,fold  (hashed names and groups)

On the other machine ``--groups <dst>/fold_table.csv`` (training and scoring)
reproduces the cluster's folds exactly: the ``fold`` column overrides the
KFold, the group and rank columns keep the single-image evaluation set
(``--test-singles``). Latents written there carry the hashed names, which the
cluster can map back with the same salt. Image headers are rewritten with
empty description, database, auxiliary-file and intent fields and no
extensions (``--no-scrub`` symlinks the sources instead, for a dry run). Prints
counts only. Copying and scrubbing ~8,000 images takes minutes: run it as a
short CPU job, not on a login node.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

from neurocausalpfn.utils.hashing import hash_filename, hash_token, load_salt

from .folds import N_FOLDS, all_folds, basenames, list_images, load_groups, modality_dirs, task_dir


def scrub_copy(src: str, dst: str) -> None:
    """Write ``src`` to ``dst`` with the identifying NIfTI header fields emptied."""
    import nibabel as nib
    import numpy as np

    img = nib.load(src)
    hdr = img.header.copy()
    for field in ("descrip", "aux_file", "db_name", "intent_name"):
        if field in hdr:
            hdr[field] = b""
    hdr.extensions.clear()
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, hdr), dst)


def fold_of_names(names: List[str], groups, n_folds: int, mode: str) -> Dict[str, int]:
    """``name -> fold`` of the test side (-1 for images never tested)."""
    out = {nm: -1 for nm in names}
    for k, (_, te) in enumerate(all_folds(len(names), n_folds, names=names, groups=groups, mode=mode)):
        for i in te:
            out[names[i]] = k
    return out


def export(data_dir: str, groups_csv: str, dst: str, salt: str, task: str = "disconnectome",
           n_folds: int = N_FOLDS, mode: str = "giles", scrub: bool = True) -> Dict[str, int]:
    groups = load_groups(groups_csv)
    if groups is None:
        sys.exit("a group table is required: the export fixes the group-aware folds")
    dirs = modality_dirs(data_dir)
    canonical = list_images(task_dir(dirs, task))
    names = basenames(canonical)
    if not names:
        sys.exit("empty listing")
    folds = fold_of_names(names, groups, n_folds, mode)
    hashed = {nm: hash_filename(nm, salt) for nm in names}
    if len(set(hashed.values())) != len(names):
        sys.exit("hash collision in the listing")

    tier = os.path.basename(os.path.normpath(data_dir))
    counts = {"images": 0, "linked": 0}
    for kind, src_dir in dirs.items():
        src_by_name = {os.path.basename(p): p for p in list_images(src_dir)}
        missing = [nm for nm in names if nm not in src_by_name]
        if missing:
            sys.exit(f"{len(missing)} files of the listing are missing from the {kind} folder")
        out_dir = os.path.join(dst, tier, kind)
        os.makedirs(out_dir, exist_ok=True)
        for nm in names:
            target = os.path.join(out_dir, hashed[nm])
            if os.path.exists(target) or os.path.islink(target):
                continue
            if scrub:
                scrub_copy(src_by_name[nm], target)
                counts["images"] += 1
            else:
                os.symlink(os.path.abspath(src_by_name[nm]), target)
                counts["linked"] += 1

    table_path = os.path.join(dst, "fold_table.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "group", "rank", "fold"])
        for nm in sorted(names, key=lambda x: hashed[x]):
            g, rank = groups[nm]
            w.writerow([hashed[nm], hash_token(g, salt), int(rank), folds[nm]])
    counts.update(n=len(names), tested=sum(1 for v in folds.values() if v >= 0),
                  never_tested=sum(1 for v in folds.values() if v < 0), n_folds=n_folds)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help='tier folder, e.g. "data/Full data"')
    ap.add_argument("--groups", required=True, help="filename,group,rank CSV (the folds' table)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--task", default="disconnectome", choices=["disconnectome", "lesion"])
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--group-mode", default="giles", choices=["giles", "strict"])
    ap.add_argument("--salt-file", default=None, help="default $HASH_SALT_FILE or ~/.cache_salt")
    ap.add_argument("--no-scrub", action="store_true", help="symlink the images instead of scrubbed copies")
    args = ap.parse_args(argv)
    c = export(args.data_dir, args.groups, args.dst, load_salt(args.salt_file), task=args.task,
               n_folds=args.n_folds, mode=args.group_mode, scrub=not args.no_scrub)
    print(f"{c['n']} images per modality, {c['tested']} tested across {c['n_folds']} folds, "
          f"{c['never_tested']} never tested; {c['images']} scrubbed copies, {c['linked']} links; "
          f"fold table: {os.path.join(args.dst, 'fold_table.csv')}")


if __name__ == "__main__":
    main()
