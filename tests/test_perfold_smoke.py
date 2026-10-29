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


def test_e1b_bernoulli_disconnectome_fold(tiny_data, tmp_path):
    """E1b (29 Oct): disconnectome channel only, its VAE on the BCE likelihood."""
    from ncpfold.catalogue import find_run
    from ncpfold.encoders import _vae_cfg

    meta = find_run("E1b").meta
    cfg = _vae_cfg(meta, "disconnectome", 50, 0, str(tmp_path / "c"), {"disconnectomes": "x", "lesions": "y"},
                   "published", None, "cpu")
    assert cfg["vae"]["disco_recon"] == "bce"
    assert "disco_recon" not in _vae_cfg(find_run("E1").meta, "disconnectome", 50, 0, str(tmp_path / "d"),
                                         {"disconnectomes": "x", "lesions": "y"}, "published", None, "cpu")["vae"]
    out = str(tmp_path / "out")
    path = run_fold("E1b", None, None, seed=0, fold=0, data_dir=tiny_data, out_root=out,
                    n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    _check(path, 3, 0, dim=50, channel="disconnectome")
    m = json.load(open(os.path.join(os.path.dirname(path), "meta.json")))
    assert m["meta"]["disco_recon"] == "bce"
    assert not os.path.exists(os.path.join(out, "E1b", "E1b", "seed0", "fold0", "lesion"))   # one channel


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


def test_tiers_on_a_trained_fold(tiny_data, tmp_path):
    """Tiers 1 and 3 (29 Oct) on the encoders of one smoke E1 fold."""
    import csv

    from ncpfold.tiers import run as tiers_run

    out = str(tmp_path / "out")
    run_fold("E1", None, None, seed=0, fold=1, data_dir=tiny_data, out_root=out,
             n_folds=3, overrides=SMOKE_OVERRIDES, device="cpu")
    var = os.path.join(out, "E1", "E1", "seed0")
    for ch in ("disco", "lesion"):
        rows = tiers_run(var, tiny_data, ch, str(tmp_path / f"t_{ch}.csv"), folds=[0, 1], device="cpu")
        assert len(rows) == 1 and rows[0]["fold"] == 1                  # fold 0 was not trained here
        r = rows[0]
        assert 0.0 <= r["dice_mean"] <= 1.0 and 0 <= r["active_dims"] <= r["zdim"] == 50
        assert r["n_test"] == len(np.load(os.path.join(var, "folds", "fold1.npz"))["te_idx"])
        with open(tmp_path / f"t_{ch}.csv") as f:
            assert len(list(csv.DictReader(f))) == 1
