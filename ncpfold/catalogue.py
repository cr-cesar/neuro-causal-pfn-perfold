"""The per-fold study catalogue: which encoders are trained, with which config.

The variants are built by the main package's own builder
(``neurocausalpfn.experiments.runner.build_runs``) from the Table-9 registry,
under the context the Phase-1 chain finally pinned (backbone resnet, lambda_Dice
0.1, 100+100 dims, ARD prior, disconnectome-only channel). Two consequences,
both deliberate:

- Every variant is defined by the same code that defined the chain, so the
  labels, grids and metadata are identical to the chain's.
- Experiments that ran before a pin was set ran under the context of their day
  (E4 trained its lesion encoders with lambda_Dice 0.5; here it uses the pinned
  0.1). Within this study every variant sees the final pinned context, which is
  what a clean re-run should do.

Excluded: E0 (the replica computes volume and per-fold NMF itself), E8 and E5c
(need the clinical CSV; wire ``--clinical-csv`` when it exists) and E12 (Phase
2, not an encoder).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from neurocausalpfn.experiments.registry import get_experiment
from neurocausalpfn.experiments.runner import RunSpec, build_runs

# The Phase-1 chain's pinned decisions (outputs/experiments/context.json,
# "manual" block) and the two rankings E11b's 2x2 grid reads.
PINNED_CONTEXT: Dict = {
    "backbone": "resnet",
    "w_dice": 0.1,
    "dims": (100, 100),
    "use_ard": True,
    "fusion_mode": "disconnectome",
    "ranking": {
        "E3": ["E3[backbone=resnet18]", "E3[backbone=cnn]",
               "E3[backbone=wide]", "E3[backbone=resnet50]"],
        "E2": ["E2[w_dice=1.0]", "E2[w_dice=0.1]", "E2[w_dice=0.5]"],
    },
}

STUDY_EIDS: List[str] = [
    "E1", "E2", "E3", "E4", "E5", "E5b", "E6", "E7a", "E7b",
    "E9a", "E9b", "E9c", "E10a", "E10b", "E10c", "E11a", "E11b",
]
EXCLUDED_EIDS = {"E0": "replica builtins", "E8": "needs clinical CSV",
                 "E5c": "needs clinical CSV", "E12": "Phase 2 (not an encoder)"}

# GPU-hours per trained encoder on an A100 (from the Phase-1 logs: a 200-epoch
# lesion or disconnectome VAE takes ~1.2 h, resnet50 ~2.3 h). The non-VAE
# trainers (100 epochs) are ESTIMATES until measured.
HOURS_PER_ENCODER = {"vae": 1.2, "vae_resnet50": 2.3, "vae_resnet18": 1.5,
                     "dmvae": 2.0, "contrastive": 2.0, "mae": 2.0, "dscm": 1.5}
PUBLISHED_BUDGET_SCALE = 32.0 / 200.0     # 32-epoch cap against the chain's 200


def study_runs(eids: Optional[List[str]] = None, context: Optional[Dict] = None) -> List[RunSpec]:
    """RunSpecs of the study, in registry order, under the pinned context."""
    ctx = dict(PINNED_CONTEXT if context is None else context)
    ctx["ranking"] = dict(ctx.get("ranking", {}))
    out: List[RunSpec] = []
    for eid in (eids or STUDY_EIDS):
        if eid in EXCLUDED_EIDS:
            raise ValueError(f"{eid} is excluded from the study: {EXCLUDED_EIDS[eid]}")
        exp = get_experiment(eid)
        specs = build_runs(exp, "full", ctx)
        for sp in specs:
            sp.meta = dict(sp.meta)
            sp.meta["eid"] = eid
        out.extend(specs)
    return out


def find_run(eid: str, run: Optional[int] = None, label: Optional[str] = None) -> RunSpec:
    specs = study_runs([eid])
    if label is not None:
        for sp in specs:
            if sp.label == label:
                return sp
        raise KeyError(f"{label!r} is not a variant of {eid}: {[s.label for s in specs]}")
    run = 0 if run is None else int(run)
    if not 0 <= run < len(specs):
        raise IndexError(f"{eid} has {len(specs)} variants; run index {run} out of range")
    return specs[run]


def encoders_of(spec: RunSpec) -> List[str]:
    """The encoders one fold of this variant trains (for the cost plan)."""
    kind = spec.kind
    if kind == "fusion_vae":
        fusion = spec.meta.get("fusion_mode", "both")
        parts = []
        if fusion in ("both", "lesion"):
            parts.append("vae")
        if fusion in ("both", "disconnectome"):
            parts.append("vae")
        bb = spec.meta.get("backbone", "resnet")
        return [f"vae_{bb}" if f"vae_{bb}" in HOURS_PER_ENCODER else "vae" for _ in parts]
    if kind == "vae_single":
        return ["vae"]
    return [kind]


def hours_for(spec: RunSpec, budget: str = "chain") -> float:
    h = sum(HOURS_PER_ENCODER[e] for e in encoders_of(spec))
    return h * (PUBLISHED_BUDGET_SCALE if budget == "published" else 1.0)
