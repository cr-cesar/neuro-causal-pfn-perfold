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


def _check(npz_path, n_folds, fold, dim, channel):
    z = np.load(npz_path, allow_pickle=False)
    tr, te = fold_indices(N_IMAGES, fold, n_folds)
    assert np.array_equal(z["tr_idx"], tr) and np.array_equal(z["te_idx"], te)
    assert z["Ztr"].shape == (len(tr), dim) and z["Zte"].shape == (len(te), dim)
    assert np.isfinite(z["Ztr"]).all() and np.isfinite(z["Zte"]).all()
    assert len(z["files"]) == N_IMAGES and str(z["channel"]) == channel
    meta = json.load(open(os.path.join(os.path.dirname(npz_path), "meta.json")))
    assert meta["dim"] == dim and meta["n_folds"] == n_folds and meta["channel"] == channel


def test_e1_fold_end_to_end(tiny_data, tmp_path):
    out = str(tmp_path / "out")
    path = run_fold("E1", None, None, seed=0, fold=1, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    var = os.path.join(out, "E1", "E1", "seed0")
    assert path == os.path.join(var, "folds", "fold1.npz")
    _check(path, 3, 1, dim=50, channel="disconnectome")            # primary = disco channel
    _check(os.path.join(var, "folds_disco", "fold1.npz"), 3, 1, 50, "disconnectome")
    _check(os.path.join(var, "folds_lesion", "fold1.npz"), 3, 1, 50, "lesion")
    _check(os.path.join(var, "folds_both", "fold1.npz"), 3, 1, 100, "both")   # lesion 50 + disco 50
    both = np.load(os.path.join(var, "folds_both", "fold1.npz"))
    les = np.load(os.path.join(var, "folds_lesion", "fold1.npz"))
    assert np.allclose(both["Zte"][:, :50], les["Zte"])                       # lesion first
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
    _check(path, 3, 0, dim=100, channel="both")       # one 2-channel VAE, ctx dims[0] = 100
    assert not os.path.exists(os.path.join(out, "E7a", "E7a", "seed0", "folds_lesion"))


def test_e7b_dmvae_fold(tiny_data, tmp_path):
    out = str(tmp_path / "out")
    path = run_fold("E7b", None, None, seed=0, fold=2, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    _check(path, 3, 2, dim=50 + 2 * 25, channel="both")   # shared 50 + private 25 x 2
