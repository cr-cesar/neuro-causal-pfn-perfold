"""End-to-end smokes on the tiny synthetic cohort (CPU, seconds): one fold of a
two-modality VAE variant (E1), of the early-fusion VAE (E7a) and of the DMVAE
(E7b). They prove the fold folders, the trainers' data-root redirection, the
full-listing encoding and the fold npz layout, not convergence."""
import json
import os

import numpy as np
import pytest

from ncpfold.folds import fold_indices, list_images, modality_dirs
from ncpfold.perfold import run_fold
from tests.conftest import N_IMAGES, SMOKE_OVERRIDES

pytest.importorskip("torch")


def _check(npz_path, n_folds, fold, dim):
    z = np.load(npz_path, allow_pickle=False)
    tr, te = fold_indices(N_IMAGES, fold, n_folds)
    assert np.array_equal(z["tr_idx"], tr) and np.array_equal(z["te_idx"], te)
    assert z["Ztr"].shape == (len(tr), dim) and z["Zte"].shape == (len(te), dim)
    assert np.isfinite(z["Ztr"]).all() and np.isfinite(z["Zte"]).all()
    assert len(z["files"]) == N_IMAGES
    meta = json.load(open(os.path.join(os.path.dirname(npz_path), "meta.json")))
    assert meta["dim"] == dim and meta["n_folds"] == n_folds


def test_e1_fold_end_to_end(tiny_data, tmp_path):
    out = str(tmp_path / "out")
    path = run_fold("E1", None, None, seed=0, fold=1, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    _check(path, 3, 1, dim=100)                       # lesion 50 + disco 50
    # both trainers ran on the fold's train side only
    dirs = modality_dirs(tiny_data)
    canonical = list_images(dirs["disconnectomes"])
    tr, _ = fold_indices(len(canonical), 1, 3)
    linked = sorted(os.listdir(os.path.join(out, "folds", "fold1", "lesions")))
    assert len(linked) == len(tr)
    assert os.path.exists(os.path.join(out, "E1", "E1", "seed0", "fold1", "lesion", "vae_lesion.pt"))
    assert os.path.exists(os.path.join(out, "E1", "E1", "seed0", "fold1", "disco", "vae_disconnectome.pt"))
    # a second call is a no-op (the fold is done)
    assert run_fold("E1", None, None, seed=0, fold=1, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu") == path


def test_e7a_early_fusion_fold(tiny_data, tmp_path):
    out = str(tmp_path / "out")
    path = run_fold("E7a", None, None, seed=0, fold=0, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    _check(path, 3, 0, dim=100)                       # one 2-channel VAE, ctx dims[0] = 100


def test_e7b_dmvae_fold(tiny_data, tmp_path):
    out = str(tmp_path / "out")
    path = run_fold("E7b", None, None, seed=0, fold=2, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    _check(path, 3, 2, dim=50 + 2 * 25)               # shared 50 + private 25 x 2
