"""Reference points (volume, NMF-50): the condition sets the simulated deficits
and the map the baselines are read from, as in the paper's code."""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from ncpfold.folds import list_images
from ncpfold.score import reference_reps, ref_inputs


def _save(path, n_on, shape=(6, 6, 6), value=1.0):
    a = np.zeros(int(np.prod(shape)), dtype=np.float32)
    a[:n_on] = value
    nib.save(nib.Nifti1Image(a.reshape(shape), np.eye(4)), str(path))


def _cohort(tmp_path, n=4, names=None):
    dirs = {k: tmp_path / k for k in ("lesions", "disconnectomes")}
    for d in dirs.values():
        d.mkdir()
    names = names or [f"img{i:02d}.nii.gz" for i in range(n)]
    for i, nm in enumerate(names):
        _save(dirs["lesions"] / nm, 3 + 3 * i)
        _save(dirs["disconnectomes"] / nm, 150, value=0.9)
    return {k: str(d) for k, d in dirs.items()}


def test_lesion_condition_loads_its_own_listing_and_labels(tmp_path):
    dirs = _cohort(tmp_path)
    disco = list_images(dirs["disconnectomes"])
    files_by_task, labels_by_task = {"disconnectome": disco}, {"disconnectome": "disco labels"}
    calls = []

    def load(files, images_dir):
        calls.append(images_dir)
        return "lesion labels"

    files, labels = ref_inputs("lesion", files_by_task, labels_by_task, dirs, load)
    assert labels == "lesion labels" and calls == [dirs["lesions"]]
    assert [os.path.dirname(f) for f in files] == [dirs["lesions"]] * 4
    assert [os.path.basename(f) for f in files] == [os.path.basename(f) for f in disco]
    # the scored task needs no loading, and a condition already loaded is kept
    assert ref_inputs("disconnectome", files_by_task, labels_by_task, dirs, load)[1] == "disco labels"
    ref_inputs("lesion", files_by_task, labels_by_task, dirs, load)
    assert calls == [dirs["lesions"]]


def test_listings_that_name_other_images_are_refused(tmp_path):
    dirs = _cohort(tmp_path)
    os.rename(os.path.join(dirs["lesions"], "img03.nii.gz"), os.path.join(dirs["lesions"], "other.nii.gz"))
    files_by_task = {"disconnectome": list_images(dirs["disconnectomes"])}
    with pytest.raises(SystemExit, match="different images"):
        ref_inputs("lesion", files_by_task, {"disconnectome": None}, dirs, lambda f, d: None)


def test_volume_is_the_conditions_own_map_and_nmf_reads_its_files(tmp_path):
    dirs = _cohort(tmp_path, n=60)                   # NMF-50 needs at least 50 training images
    lesion_files = list_images(dirs["lesions"])
    labels = pd.DataFrame({"vol": [3 + 3 * i for i in range(60)]})
    reps = reference_reps(lesion_files, labels, with_volume=True, with_nmf=True)
    np.testing.assert_allclose(reps["volume"][:, 0], labels["vol"] / labels["vol"].max())
    tr, te = np.arange(52), np.arange(52, 60)
    Ztr, Zte = reps["nmf50_perfold"](tr, te)
    assert Ztr.shape == (52, 50) and Zte.shape == (8, 50)
    # the masks differ image to image, the disconnectomes do not
    Dtr, _ = reference_reps(list_images(dirs["disconnectomes"]), labels, False, True)["nmf50_perfold"](tr, te)
    assert np.ptp(Dtr, axis=0).max() < 0.1 * np.ptp(Ztr, axis=0).max()
    assert reference_reps(lesion_files, labels, False, False) == {}
