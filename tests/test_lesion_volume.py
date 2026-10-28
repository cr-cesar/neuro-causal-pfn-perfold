import os

import nibabel as nib
import numpy as np
import pytest

from ncpfold.score import _lesion_masks, _lesion_volumes, _nmf_perfold


def _save(path, n_on, shape=(6, 6, 6), value=1.0):
    a = np.zeros(int(np.prod(shape)), dtype=np.float32)
    a[:n_on] = value
    nib.save(nib.Nifti1Image(a.reshape(shape), np.eye(4)), str(path))


def test_volume_counts_the_lesion_mask_not_the_disconnectome(tmp_path):
    dirs = {k: tmp_path / k for k in ("lesions", "disconnectomes")}
    for d in dirs.values():
        d.mkdir()
    names = ["a.nii.gz", "b.nii.gz", "c.nii.gz"]
    for i, nm in enumerate(names):
        _save(dirs["lesions"] / nm, 5 * (i + 1))
        _save(dirs["disconnectomes"] / nm, 50 + i, value=0.9)
    v = _lesion_volumes(names, {k: str(d) for k, d in dirs.items()})
    np.testing.assert_array_equal(v, [5.0, 10.0, 15.0])


def test_volume_needs_a_mask_per_image(tmp_path):
    (tmp_path / "lesions").mkdir()
    _save(tmp_path / "lesions" / "a.nii.gz", 3)
    with pytest.raises(KeyError):
        _lesion_volumes(["a.nii.gz", "b.nii.gz"], {"lesions": str(tmp_path / "lesions")})


def test_nmf_lesion_reads_the_masks_in_the_listing_order(tmp_path):
    dirs = {k: tmp_path / k for k in ("lesions", "disconnectomes")}
    for d in dirs.values():
        d.mkdir()
    names = [f"img{i:02d}.nii.gz" for i in range(60)]     # NMF-50 needs at least 50 training images
    for i, nm in enumerate(names):
        _save(dirs["lesions"] / nm, 3 + 3 * i)
        _save(dirs["disconnectomes"] / nm, 150, value=0.9)   # identical maps
    sdirs = {k: str(d) for k, d in dirs.items()}
    paths = _lesion_masks(list(reversed(names)), sdirs)
    assert [p.replace("\\", "/").split("/")[-2:] for p in paths] == [["lesions", nm] for nm in reversed(names)]
    tr, te = np.arange(52), np.arange(52, 60)
    Ztr, Zte = _nmf_perfold(_lesion_masks(names, sdirs))(tr, te)
    assert Ztr.shape == (52, 50) and Zte.shape == (8, 50)
    # the masks differ image to image, the disconnectomes do not: the factor
    # scores spread across images for the masks only
    Dtr, _ = _nmf_perfold([os.path.join(sdirs["disconnectomes"], nm) for nm in names])(tr, te)
    assert np.ptp(Dtr, axis=0).max() < 0.1 * np.ptp(Ztr, axis=0).max()
