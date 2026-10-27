"""Step 0 of the 27 Oct plan: how much improvement is possible on a representation.

Three descriptive reference points per deficit and fold, on the same folds,
slices and test sets as the replica scorer (they select nothing, so they may
be computed on the tested patients and on the validation extras alike):

- latent oracle: a regressor fitted on the TRUE effect (tau in {-1, 0, +1})
  of the fold's training patients from the latents, scored on the test
  patients; gradient boosting and ridge. The best any estimator could read
  from these latents when the effect itself is the target.
- true-anatomy oracle: the same regressor on the lesion's coverage of the two
  halves of the deficit's network, read from the cohort's anatomy cache (the
  quantities the trial rule thresholds). The best achievable with perfect
  information about the lesion.
- probe: cross-validated AUC (grouped by patient) of a logistic regression
  predicting "covers half A above threshold" and "half B" from the latents:
  whether the decisive information survives the encoding.

    python -m ncpfold.headroom --reps outputs_perfold_ws/E1/E1/seed0/folds --data-dir "export/Full data" \\
        --groups export/fold_table.csv --test-singles --cache outputs/prior_cache/uclh_disco_x.npz \\
        --modality receptor --out status/headroom/oracles_2828_receptor.csv

The cache's ``files`` must name the same images as the listing (real names on
the cluster, hashed names in an export made with the same salt). Aggregates
only are written.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from neurocausalpfn.prior import giles_replica as gr

from .folds import all_folds, list_images, load_groups, modality_dirs, task_dir
from .score import cached_labels, extras_folds, extras_mask, fold_lookup, rep_meta, restrict_test, singles_mask


def truth_from_labels(labels, name: str):
    """Susceptibility mask and tau in {-1, 0, +1} per row of ``labels`` for
    deficit ``name`` (W1 hit -> +1, W0 hit -> -1, both -> 0)."""
    w0 = labels[f"{name}_W0"].to_numpy() == 1
    w1 = labels[f"{name}_W1"].to_numpy() == 1
    mask = w0 | w1
    tau = np.where(w0 & w1, 0.0, np.where(w1, 1.0, -1.0))
    return mask, tau


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def oracle_rows(Ztr, Zte, Atr, Ate, tau_tr, tau_te, hits_tr, groups_tr, seed: int = 0) -> Dict[str, float]:
    """One deficit x fold: oracles and probe from already sliced arrays.
    ``Atr/Ate`` are the anatomy features (coverage of the two halves) or None;
    ``hits_tr`` is a dict name -> 0/1 labels for the probe; ``groups_tr`` the
    patient id per training row (grouped cross-validation)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    out: Dict[str, float] = {"n_train": int(len(tau_tr)), "n_test": int(len(tau_te)),
                             "share_zero_test": float(np.mean(tau_te == 0)) if len(tau_te) else float("nan")}
    if len(tau_te) == 0 or len(tau_tr) < 10:
        return out
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    out["pehe_latent_hgb"] = _rmse(np.clip(hgb.fit(Ztr, tau_tr).predict(Zte), -1, 1), tau_te)
    out["pehe_latent_ridge"] = _rmse(np.clip(Ridge(1.0).fit(Ztr, tau_tr).predict(Zte), -1, 1), tau_te)
    if Atr is not None:
        hgb_a = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
        out["pehe_anatomy_hgb"] = _rmse(np.clip(hgb_a.fit(Atr, tau_tr).predict(Ate), -1, 1), tau_te)
    n_groups = len(np.unique(groups_tr))
    for key, y in hits_tr.items():
        y = np.asarray(y, int)
        if len(np.unique(y)) < 2 or n_groups < 3:
            out[f"auc_probe_{key}"] = float("nan"); continue
        cv = GroupKFold(n_splits=min(5, n_groups))
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Ztr, y, cv=cv, groups=groups_tr,
                              method="predict_proba")[:, 1]
        out[f"auc_probe_{key}"] = float(roc_auc_score(y, p))
    return out


def anatomy_features(cache_path: Optional[str], names: List[str], modality: str) -> Optional[np.ndarray]:
    """[n, K, 2] coverage of the two halves of every network of ``modality``
    for the listing, from a cohort anatomy cache; None without a cache."""
    if not cache_path:
        return None
    with np.load(cache_path, allow_pickle=True) as z:
        files = [os.path.basename(str(f)) for f in np.asarray(z["files"]).ravel()]
        key = f"overlaps_{modality}"
        ov = np.asarray(z[key] if key in z.files else z["overlaps"], dtype=np.float32)
    pos = {f: i for i, f in enumerate(files)}
    missing = [nm for nm in names if nm not in pos]
    if missing:
        sys.exit(f"{len(missing)} listing images are missing from the anatomy cache")
    return ov[[pos[nm] for nm in names]]


def run(rep_dir: str, data_dir: str, atlas_dir: str, out: str, modality: str = "receptor",
        groups_csv: Optional[str] = None, test_singles: bool = False, test_extras: bool = False,
        cache: Optional[str] = None, deficits=None, seed: int = 0):
    import pandas as pd

    meta = rep_meta(rep_dir)
    task = meta.get("task", "disconnectome")
    groups = load_groups(groups_csv)
    dirs = modality_dirs(data_dir)
    images_dir = task_dir(dirs, task)
    files = list_images(images_dir)
    names = [os.path.basename(f) for f in files]
    pairs = gr.load_atlas_pairs(atlas_dir, modality)
    labels = cached_labels(files, pairs, images_dir, modality, os.path.join(os.path.dirname(out) or ".", "labels"))
    n_folds = int(meta.get("n_folds", 10))
    gmode = meta.get("group_mode") or None
    folds = all_folds(len(files), n_folds, names=names, groups=groups, mode=gmode or "giles")
    if test_singles:
        folds = restrict_test(folds, singles_mask(names, groups))
    if test_extras:
        folds = extras_folds(folds, extras_mask(names, groups), names, groups)
    lookup = fold_lookup(rep_dir, len(files))
    anat = anatomy_features(cache, names, modality)
    patient = np.array([groups[nm][0] if groups is not None and nm in groups else nm for nm in names])

    rows = []
    for d in (deficits or sorted(pairs)):
        name = gr.DEFICIT_NAMES[d - 1]
        mask, tau = truth_from_labels(labels, name)
        for k, (tr, te) in enumerate(folds):
            Ztr, Zte = lookup(tr, te)
            tr_s, te_s = tr[mask[tr]], te[mask[te]]
            if len(te_s) == 0:
                continue
            itr = np.isin(tr, tr_s); ite = np.isin(te, te_s)
            A_tr = anat[tr_s, d - 1, :] if anat is not None else None
            A_te = anat[te_s, d - 1, :] if anat is not None else None
            hits = {"W1": labels[f"{name}_W1"].to_numpy()[tr_s], "W0": labels[f"{name}_W0"].to_numpy()[tr_s]}
            r = oracle_rows(Ztr[itr], Zte[ite], A_tr, A_te, tau[tr_s], tau[te_s], hits, patient[tr_s], seed=seed)
            rows.append({"deficit": d, "name": name, "fold": k, "modality": modality, **r})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    cols = [c for c in df.columns if c.startswith("pehe_") or c.startswith("auc_") or c == "share_zero_test"]
    summ = df.groupby("deficit")[cols].mean()
    summ.loc["mean"] = summ.mean()
    summ.to_csv(out.replace(".csv", "_summary.csv"))
    print(f"{os.path.basename(rep_dir)} | {modality} | {'extras' if test_extras else 'singles' if test_singles else 'all'}"
          f" | {len(df)} deficit x fold cells")
    print(summ.loc[["mean"]].round(3).to_string())
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", required=True, help="one representation folder (fold*.npz + meta.json)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--atlas-dir", default="data/atlases")
    ap.add_argument("--modality", default="receptor", choices=["receptor", "genetics"])
    ap.add_argument("--groups", default=None)
    ap.add_argument("--test-singles", action="store_true")
    ap.add_argument("--test-extras", action="store_true")
    ap.add_argument("--cache", default=None, help="cohort anatomy cache with overlaps_<modality> (true-anatomy oracle)")
    ap.add_argument("--deficits", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.reps, a.data_dir, a.atlas_dir, a.out, modality=a.modality, groups_csv=a.groups,
        test_singles=a.test_singles, test_extras=a.test_extras, cache=a.cache, deficits=a.deficits, seed=a.seed)


if __name__ == "__main__":
    main()
