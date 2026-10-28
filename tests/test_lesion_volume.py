import nibabel as nib
import numpy as np
import pytest

from ncpfold.score import _lesion_volumes


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
