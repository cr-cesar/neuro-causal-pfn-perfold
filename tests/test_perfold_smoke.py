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


def test_local_runner_trains_a_fold_of_the_hashed_export(tiny_data, tmp_path):
    """export -> plan --emit-local -> run_jobs.sh (main repo) -> perfold_train_local.sh:
    the workstation path, end to end on the tiny cohort."""
    import shutil
    import subprocess
    import sys

    import neurocausalpfn
    from neurocausalpfn.utils.hashing import load_salt

    from ncpfold.export_hashed import export
    from ncpfold.plan import main as plan_main

    main_root = os.path.abspath(os.path.join(os.path.dirname(neurocausalpfn.__file__), "..", ".."))
    runner = os.path.join(main_root, "scripts", "run_jobs.sh")
    if not os.path.exists(runner):
        pytest.skip("main repository scripts not available")
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    names = sorted(os.listdir(os.path.join(tiny_data, "disconnectomes")))
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text("filename,group,rank\n" + "".join(f"{n},g{i},0\n" for i, n in enumerate(names)))
    dst = tmp_path / "export"
    export(tiny_data, str(groups_csv), str(dst), load_salt(str(tmp_path / "salt")), n_folds=3, scrub=False)
    data = str(dst / os.path.basename(tiny_data)); table = str(dst / "fold_table.csv"); out = str(tmp_path / "out")
    # a checkout of this repo with the runner scripts (outputs land there)
    work = tmp_path / "perfold"; (work / "qsub").mkdir(parents=True)
    shutil.copy(os.path.join(here, "qsub", "perfold_train_local.sh"), work / "qsub")
    venv = tmp_path / "env" / "bin"; venv.mkdir(parents=True)
    os.symlink(sys.executable, venv / "python")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plan_main(["--eids", "E7a", "--seeds", "1", "--n-folds", "3", "--emit-local",
                   "--data-dir", data, "--out-root", out, "--groups", table])
    lines = [ln for ln in buf.getvalue().splitlines() if "FOLD=2 " in ln]
    assert len(lines) == 1
    overrides = json.dumps(SMOKE_OVERRIDES).replace('"', '\\"')
    jobs = tmp_path / "jobs.txt"
    jobs.write_text(f'N_FOLDS=3 OVERRIDES="{overrides}" ' + lines[0] + "\n")
    r = subprocess.run(["bash", runner, str(jobs)], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "WORKDIR": str(work), "VENV": str(tmp_path / "env"),
                            "PYTHONPATH": here + os.pathsep + os.environ.get("PYTHONPATH", "")})
    log = (work / "logs" / "1_e7ar0s0f2.log").read_text() if (work / "logs").exists() else ""
    assert r.returncode == 0 and "ok 1" in r.stdout, r.stdout + r.stderr + log
    z = np.load(os.path.join(out, "E7a", "E7a", "seed0", "folds", "fold2.npz"))
    assert str(z["group_mode"]) == "giles" and len(z["files"]) == N_IMAGES
    assert all(len(str(f)) == 23 for f in z["files"])            # 16 hex digits + .nii.gz: no raw name
    assert "group table:" in log and "fold 2" in log
