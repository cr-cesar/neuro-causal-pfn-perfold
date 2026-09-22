"""Train one variant's encoder(s) on a fold's train side and encode the whole
canonical listing with the frozen result.

Each trainer of the main package is called through its public ``run_*`` entry
point with its own ``full_config()``, exactly as the Phase-1 runner does, with
two differences only: the data roots point at the fold's symlink folders, and
nothing is exported by the trainer (the encoding is done here, over the full
listing, so both sides of the fold are encoded by the same frozen model).

Which weights encode: for the VAE family the best-validation checkpoint the
trainer wrote (``vae_<rep>.pt``, the same file the Phase-1 replica
certification exported from); for the other trainers their single checkpoint
(final epoch, the only one they keep).
"""
from __future__ import annotations

import copy
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocausalpfn.data.nifti_dataset import (LesionMaskDataset,
                                               PairedLesionDisconnectomeDataset)
from neurocausalpfn.experiments.runner import RunSpec
from neurocausalpfn.utils.runtime import resolve_device

from .folds import align_to_canonical, basenames

PUBLISHED_VAE_EPOCHS = 32          # the paper's cap (min 16 / max 32, early stop 4)
CHAIN_VAE_EPOCHS = 200


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def apply_budget(cfg: Dict, budget: str) -> Dict:
    """``chain`` keeps every trainer's own epochs (the only change of the study
    is then the fold protocol); ``published`` caps the VAEs at 32 epochs and
    scales the other trainers by the same 32/200 ratio."""
    if budget == "chain":
        return cfg
    if budget != "published":
        raise ValueError(f"budget must be 'chain' or 'published', got {budget!r}")
    if "vae" in cfg:
        cfg["vae"]["epochs"] = min(int(cfg["vae"]["epochs"]), PUBLISHED_VAE_EPOCHS)
    if "train" in cfg and "epochs" in cfg["train"]:
        cfg["train"]["epochs"] = max(1, int(round(cfg["train"]["epochs"]
                                                  * PUBLISHED_VAE_EPOCHS / CHAIN_VAE_EPOCHS)))
    return cfg


def apply_overrides(cfg: Dict, overrides: Optional[Dict]) -> Dict:
    """Smoke/test knobs (resolution, epochs, batch_size, channels, device...).
    Never used on the cluster runs."""
    if not overrides:
        return cfg
    if "resolution" in overrides:
        cfg["data"]["resolution"] = [int(v) for v in overrides["resolution"]]
    for key in ("epochs", "batch_size"):
        if key in overrides:
            for sec in ("vae", "train"):
                if sec in cfg and key in cfg[sec]:
                    cfg[sec][key] = overrides[key]
    if "channels" in overrides:
        for sec in ("vae", "model"):
            if sec in cfg and "channels" in cfg[sec]:
                cfg[sec]["channels"] = [int(c) for c in overrides["channels"]]
    for key in ("device", "amp", "num_workers"):
        if key in overrides:
            cfg[key] = overrides[key]
    if "model" in overrides and "model" in cfg:
        cfg["model"].update(overrides["model"])
    return cfg


def _roots(cfg: Dict, fold_dirs: Dict[str, str], primary: str) -> None:
    cfg["data"]["n_synth"] = 0
    cfg["data"]["root"] = fold_dirs[primary]
    cfg["data"]["lesion_root"] = fold_dirs["lesions"]
    cfg["data"]["disconnectome_root"] = fold_dirs["disconnectomes"]


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_dataset(fn: Callable, dataset, batch_size: int, device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codes = []
    for batch in loader:
        items = list(batch) if isinstance(batch, (list, tuple)) else [batch]
        codes.append(fn([t.to(device) for t in items]).float().cpu().numpy())
    return np.concatenate(codes, axis=0)


def _lesion_dataset(full_dirs, kind: str, resolution, seed: int, binarize: bool, **kw):
    ds = LesionMaskDataset(root=full_dirs[kind], in_shape=tuple(resolution),
                           n_synth=0, seed=seed, binarize=binarize, **kw)
    if ds.synthetic or len(ds) == 0:
        raise FileNotFoundError(f"no images under {full_dirs[kind]}")
    return ds, basenames(ds.paths)


def _paired_dataset(full_dirs, resolution, seed: int, **kw):
    ds = PairedLesionDisconnectomeDataset(lesion_root=full_dirs["lesions"],
                                          disconnectome_root=full_dirs["disconnectomes"],
                                          in_shape=tuple(resolution), n_synth=0, seed=seed, **kw)
    if ds.synthetic or len(ds) == 0:
        raise FileNotFoundError("no paired images under the full data folders")
    return ds, basenames([lp for lp, _ in ds.pairs])


def _release(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Trainer adapters, one per RunSpec.kind
# --------------------------------------------------------------------------- #
def _vae_cfg(meta: Dict, rep: str, zdim: int, seed: int, out_dir: str,
             fold_dirs: Dict[str, str], budget: str, overrides, device: str) -> Dict:
    from neurocausalpfn.train.train_vae import full_config

    if meta.get("use_daft"):
        raise NotImplementedError("DAFT conditioning (E8/E5c) needs the clinical CSV; not wired yet")
    cfg = full_config()
    cfg["seed"] = seed
    cfg["out_dir"] = out_dir
    cfg["export"] = False
    cfg["representation"] = rep
    cfg["device"] = device
    cfg["vae"]["zdim"] = int(zdim)
    cfg["vae"]["backbone"] = meta.get("backbone", "resnet")
    cfg["vae"]["w_dice"] = 1.0 if rep == "disconnectome" else float(meta.get("w_dice", 1.0))
    cfg["vae"]["use_daft"] = False
    cfg["vae"]["use_ard"] = bool(meta.get("use_ard", False))
    cfg["vae"]["use_pns"] = bool(meta.get("use_pns", False))
    cfg["vae"]["lambda_pns"] = float(meta.get("lambda_pns", 0.1))
    _roots(cfg, fold_dirs, "lesions" if rep in ("lesion", "early_fusion") else "disconnectomes")
    return apply_overrides(apply_budget(cfg, budget), overrides)


def _run_vae_best(cfg: Dict, rep: str, device):
    from neurocausalpfn.train.train_vae import run_vae

    model, _ = run_vae(cfg)
    best = os.path.join(cfg["out_dir"], f"vae_{rep}.pt")
    if os.path.exists(best):
        state = torch.load(best, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state)
    return model.to(device).eval()


def _fusion_vae(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    meta = spec.meta
    fusion = meta.get("fusion_mode", "both")
    reps: List = []
    if fusion in ("both", "lesion"):
        reps.append(("lesion", "lesions", os.path.join(out_dir, "lesion"), meta["d_lesion"]))
    if fusion in ("both", "disconnectome"):
        reps.append(("disconnectome", "disconnectomes", os.path.join(out_dir, "disco"),
                     meta.get("d_disco", meta["d_lesion"])))
    parts = []
    for rep, kind, sub, zdim in reps:                    # lesion first, then disco (chain order)
        cfg = _vae_cfg(meta, rep, zdim, seed, sub, fold_dirs, budget, overrides, device)
        model = _run_vae_best(cfg, rep, device)
        ds, names = _lesion_dataset(full_dirs, kind, cfg["data"]["resolution"], seed,
                                    binarize=(rep == "lesion"))
        Z = encode_dataset(lambda items: model.encode_mean(items[0]), ds,
                           cfg["vae"]["batch_size"], device)
        parts.append(align_to_canonical(Z, names, canonical))
        _release(model)
    return np.concatenate(parts, axis=1)


def _vae_single(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    meta = spec.meta
    cfg = _vae_cfg(meta, "early_fusion", meta["d_lesion"], seed, out_dir, fold_dirs,
                   budget, overrides, device)
    model = _run_vae_best(cfg, "early_fusion", device)
    ds, names = _paired_dataset(full_dirs, cfg["data"]["resolution"], seed, stack_channels=True)
    Z = encode_dataset(lambda items: model.encode_mean(items[0]), ds,
                       cfg["vae"]["batch_size"], device)
    _release(model)
    return align_to_canonical(Z, names, canonical)


def _dmvae(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    from neurocausalpfn.train.train_dmvae import full_config, run_dmvae

    cfg = full_config()
    cfg["seed"], cfg["out_dir"], cfg["export"], cfg["device"] = seed, out_dir, False, device
    cfg["model"]["shared_dim"] = int(spec.meta["shared_dim"])
    cfg["model"]["private_dim"] = int(spec.meta["private_dim"])
    _roots(cfg, fold_dirs, "lesions")
    cfg = apply_overrides(apply_budget(cfg, budget), overrides)
    model, _ = run_dmvae(cfg)
    model = model.to(device).eval()
    ds, names = _paired_dataset(full_dirs, cfg["data"]["resolution"], seed, stack_channels=False)
    Z = encode_dataset(lambda items: model.encode_z(items[0], items[1]), ds,
                       cfg["train"]["batch_size"], device)
    _release(model)
    return align_to_canonical(Z, names, canonical)


def _contrastive(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    from neurocausalpfn.train.train_contrastive import full_config, run_contrastive

    cfg = full_config()
    cfg["seed"], cfg["out_dir"], cfg["export"], cfg["device"] = seed, out_dir, False, device
    if spec.meta.get("no_recon"):
        cfg["model"]["recon"] = False
    _roots(cfg, fold_dirs, "lesions")
    cfg = apply_overrides(apply_budget(cfg, budget), overrides)
    model, _ = run_contrastive(cfg)
    model = model.to(device).eval()
    ds, names = _paired_dataset(full_dirs, cfg["data"]["resolution"], seed, stack_channels=False)
    Z = encode_dataset(lambda items: model.encode_z(items[0], items[1]), ds,
                       cfg["train"]["batch_size"], device)
    _release(model)
    return align_to_canonical(Z, names, canonical)


def _mae(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    from neurocausalpfn.train.train_mae import full_config, run_mae

    cfg = full_config()
    cfg["seed"], cfg["out_dir"], cfg["export"], cfg["device"] = seed, out_dir, False, device
    cfg["model"]["mask_ratio"] = float(spec.meta.get("mask_ratio", 0.75))
    cfg["train"]["lesion_weight"] = float(spec.meta.get("lesion_weight", 5.0))
    _roots(cfg, fold_dirs, "lesions")
    cfg = apply_overrides(apply_budget(cfg, budget), overrides)
    model, _ = run_mae(cfg)
    model = model.to(device).eval()
    ds, names = _lesion_dataset(full_dirs, "lesions", cfg["data"]["resolution"], seed, binarize=True)
    Z = encode_dataset(lambda items: model.encode_z(items[0]), ds, cfg["train"]["batch_size"], device)
    _release(model)
    return align_to_canonical(Z, names, canonical)


def _dscm(spec, seed, fold_dirs, full_dirs, canonical, out_dir, budget, overrides, device):
    from neurocausalpfn.train.train_dscm import full_config, run_dscm

    cfg = full_config()
    cfg["seed"], cfg["out_dir"], cfg["export"], cfg["device"] = seed, out_dir, False, device
    cfg["model"]["multi_env"] = bool(spec.meta.get("multi_env", False))
    cfg["model"]["use_ard"] = bool(spec.meta.get("use_ard", False))
    _roots(cfg, fold_dirs, "lesions")
    cfg = apply_overrides(apply_budget(cfg, budget), overrides)
    model, _ = run_dscm(cfg)
    model = model.to(device).eval()
    ds, names = _lesion_dataset(full_dirs, "lesions", cfg["data"]["resolution"], seed,
                                binarize=True, with_clinical=True)
    Z = encode_dataset(lambda items: model.encode_z(items[0]), ds, cfg["train"]["batch_size"], device)
    _release(model)
    return align_to_canonical(Z, names, canonical)


ADAPTERS = {"fusion_vae": _fusion_vae, "vae_single": _vae_single, "dmvae": _dmvae,
            "contrastive": _contrastive, "mae": _mae, "dscm": _dscm}


def train_and_encode(spec: RunSpec, seed: int, fold_dirs: Dict[str, str],
                     full_dirs: Dict[str, str], canonical_names: Sequence[str],
                     out_dir: str, budget: str = "chain",
                     overrides: Optional[Dict] = None, device: str = "auto") -> np.ndarray:
    """Train ``spec`` on the fold's train side and return Z for the whole
    canonical listing, ``[n_canonical, d]``, rows in canonical order."""
    if spec.kind not in ADAPTERS:
        raise ValueError(f"no per-fold adapter for run kind {spec.kind!r} ({spec.label})")
    dev = resolve_device({"device": device})
    os.makedirs(out_dir, exist_ok=True)
    Z = ADAPTERS[spec.kind](spec, seed, dict(fold_dirs), dict(full_dirs), list(canonical_names),
                            out_dir, budget, copy.deepcopy(overrides) if overrides else None, dev)
    if Z.shape[0] != len(canonical_names):
        raise RuntimeError(f"encoded {Z.shape[0]} rows for {len(canonical_names)} files")
    return Z.astype(np.float32)
