"""Tiers 1 and 3 of the design for the per-fold encoders (29 Oct): the
reconstruction fidelity gate (Dice >= 0.70) and the latent structure (active
dimensions, KL > 0.01; IOSS), measured fold by fold on each fold's TEST images
with the encoder trained on the other folds.

The metrics are the main package's (``experiments.artifacts._binary_dice``,
``eval.latent_quality.per_dim_kl`` / ``active_dimensions`` / ``ioss``, the
thresholds of ``experiments.registry.TIER_GATES``). The chain computed them for
the frozen encoder; this applies them to the per-fold encoders of the benchmark.
The disconnectome is a continuous map: its Dice binarises both the map and the
reconstruction at 0.5, the threshold the trials use for enrolment. Counts and
means only.

    python -m ncpfold.tiers --variant-dir outputs_perfold_bcbenc/E1/E1/seed0 \\
        --data-dir "data/Full data" --channel disco --out tiers_E1_seed0_disco.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np

CHANNELS = {"disco": ("disconnectomes", "disconnectome", "folds_disco"),
            "lesion": ("lesions", "lesion", "folds_lesion")}


def _fold_npz(variant_dir: str, folder: str, k: int) -> str:
    for f in (folder, "folds"):
        p = os.path.join(variant_dir, f, f"fold{k}.npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no fold{k}.npz under {variant_dir}")


def load_encoder(ckpt_path: str, device: str):
    import torch

    from neurocausalpfn.vae.conv3d_vae import ConvVAE3D

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = ConvVAE3D(in_channels=int(ck.get("in_channels", 1)), zdim=int(cfg["vae"]["zdim"]),
                      in_shape=tuple(cfg["data"]["resolution"]), channels=tuple(cfg["vae"]["channels"]),
                      backbone=ck.get("backbone", cfg["vae"].get("backbone", "resnet")),
                      use_daft=False, n_clinical=0, use_ard=bool(ck.get("use_ard", False)))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def fold_metrics(model, dataset, device: str, batch_size: int = 4) -> Dict[str, float]:
    """Mean per-image Dice of the reconstruction (both binarised at 0.5) and the
    KL diagnostics of the posterior over the given images."""
    import torch
    from torch.utils.data import DataLoader

    from neurocausalpfn.eval.latent_quality import active_dimensions, ioss, per_dim_kl
    from neurocausalpfn.experiments.artifacts import _binary_dice

    mus, lvs, dices = [], [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            mu, logvar = model.enc(x)
            logits = model.dec(mu)[:, 0:1]
            target = (x[:, 0:1] > 0.5).float()
            for i in range(x.shape[0]):
                dices.append(_binary_dice(logits[i:i + 1], target[i:i + 1]))
            mus.append(mu.cpu().numpy()); lvs.append(logvar.cpu().numpy())
    Z, LV = np.concatenate(mus), np.concatenate(lvs)
    pdk = per_dim_kl(Z, LV, None)
    return {"n_test": int(len(Z)), "dice_mean": float(np.mean(dices)), "dice_median": float(np.median(dices)),
            "share_dice_ge_070": float(np.mean(np.asarray(dices) >= 0.70)),
            "active_dims": int(active_dimensions(pdk, threshold=0.01)), "zdim": int(Z.shape[1]),
            "kl_mean": float(np.mean(pdk)), "ioss": float(ioss(Z, seed=0))}


def run(variant_dir: str, data_dir: str, channel: str, out: str, folds: List[int] = None,
        device: str = "auto", batch_size: int = 4) -> List[Dict]:
    import torch
    from torch.utils.data import Subset

    from neurocausalpfn.data.nifti_dataset import LesionMaskDataset

    from .folds import basenames

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sub, rep, folder = CHANNELS[channel]
    rows, ds, names = [], None, None
    for k in (folds if folds is not None else range(10)):
        ck = os.path.join(variant_dir, f"fold{k}", channel, f"vae_{rep}.pt")
        if not os.path.exists(ck):
            print(f"fold {k}: no checkpoint, skipped"); continue
        model, cfg = load_encoder(ck, device)
        if ds is None:
            ds = LesionMaskDataset(root=os.path.join(data_dir, sub), in_shape=tuple(cfg["data"]["resolution"]),
                                   n_synth=0, seed=0, binarize=(channel == "lesion"))
            names = basenames(ds.paths)
        z = np.load(_fold_npz(variant_dir, folder, k), allow_pickle=False)
        stored = [os.path.basename(str(f)) for f in z["files"]]
        if stored != list(names):
            raise SystemExit(f"fold {k}: the image listing differs from the one the fold was encoded on")
        te = np.asarray(z["te_idx"], int)
        m = fold_metrics(model, Subset(ds, te.tolist()), device, batch_size)
        m.update({"fold": k, "channel": channel, "dice_gate_070": bool(m["dice_mean"] >= 0.70)})
        rows.append(m)
        print(f"fold {k}: Dice {m['dice_mean']:.3f}, active dims {m['active_dims']}/{m['zdim']}, n {m['n_test']}")
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with open(out[:-4] + "_meta.json", "w") as f:
            json.dump({"variant_dir": variant_dir, "channel": channel, "dice_binarised_at": 0.5,
                       "active_dim_kl_threshold": 0.01, "evaluated_on": "each fold's test images"}, f, indent=1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", required=True, help="<out-root>/<eid>/<label>/seed<S>")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--channel", default="disco", choices=sorted(CHANNELS))
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.variant_dir, a.data_dir, a.channel, a.out, a.folds, a.device, a.batch_size)


if __name__ == "__main__":
    main()
