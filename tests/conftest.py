"""Shared fixtures: a tiny synthetic paired cohort written as NIfTI files, so the
per-fold pipeline runs end to end on CPU in seconds. Also keeps the numeric
libraries at 2 threads on cluster login nodes (see the main repo's conftest)."""
import os
import re
import socket

import numpy as np
import pytest


def _on_login_node() -> bool:
    if os.environ.get("NEUROCAUSAL_LOGIN_SAFE"):
        return True
    return re.match(r"login\d+", socket.gethostname() or "") is not None


if _on_login_node():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    try:
        import torch

        torch.set_num_threads(2)
    except Exception:
        pass

SHAPE = (24, 28, 24)
N_IMAGES = 12
SMOKE_OVERRIDES = {"resolution": list(SHAPE), "epochs": 1, "batch_size": 4,
                   "channels": [8, 8, 8, 8, 8], "device": "cpu", "amp": False, "num_workers": 0}


def _field(rng, shape):
    zz, yy, xx = np.indices(shape)
    c = [rng.integers(int(0.3 * s), int(0.7 * s)) for s in shape]
    r = [rng.integers(3, max(4, int(0.25 * s))) for s in shape]
    e = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2
    return np.clip(1.5 - e, 0.0, 1.0).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """``data_dir`` with lesions/ (binary) and disconnectomes/ (continuous),
    12 paired files named lesion{id}_{age}_{sex}.nii.gz."""
    nib = pytest.importorskip("nibabel")
    root = tmp_path_factory.mktemp("tiny")
    les_dir, dis_dir = root / "lesions", root / "disconnectomes"
    les_dir.mkdir(); dis_dir.mkdir()
    rng = np.random.default_rng(0)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    for i in range(N_IMAGES):
        soft = _field(rng, SHAPE)
        name = f"lesion{i:04d}_{50 + i}_{'M' if i % 2 else 'F'}.nii.gz"
        nib.save(nib.Nifti1Image((soft > 0.5).astype(np.float32), affine), str(les_dir / name))
        nib.save(nib.Nifti1Image(soft, affine), str(dis_dir / name))
    return str(root)
