import os

import numpy as np
import pytest
from sklearn.model_selection import KFold

from ncpfold.folds import (align_to_canonical, all_folds, fold_indices, list_images,
                           make_fold_dirs, modality_dirs)


def test_folds_are_the_replicas_kfold():
    n = 4119
    ref = list(KFold(n_splits=10, shuffle=True, random_state=0).split(np.arange(n)))
    for k, (tr, te) in enumerate(all_folds(n, 10)):
        assert np.array_equal(tr, ref[k][0]) and np.array_equal(te, ref[k][1])
    tr, te = fold_indices(n, 3, 10)
    assert len(tr) + len(te) == n and not set(tr) & set(te)
    with pytest.raises(ValueError):
        fold_indices(n, 10, 10)


def test_fold_dirs_hold_only_the_train_side(tiny_data, tmp_path):
    dirs = modality_dirs(tiny_data)
    canonical = list_images(dirs["disconnectomes"])
    info = make_fold_dirs(str(tmp_path / "folds"), dirs, canonical, fold=1, n_folds=3)
    tr, te = info["tr_idx"], info["te_idx"]
    names = info["files"]
    for kind in ("lesions", "disconnectomes"):
        linked = sorted(os.listdir(info["dirs"][kind]))
        assert linked == sorted(names[i] for i in tr)
        assert all(os.path.islink(os.path.join(info["dirs"][kind], f)) for f in linked)
        assert not any(names[i] in linked for i in te)
    # idempotent: a second call leaves the same links
    again = make_fold_dirs(str(tmp_path / "folds"), dirs, canonical, fold=1, n_folds=3)
    assert sorted(os.listdir(again["dirs"]["lesions"])) == sorted(os.listdir(info["dirs"]["lesions"]))
    # a different fold in the same root does not disturb fold 1
    make_fold_dirs(str(tmp_path / "folds"), dirs, canonical, fold=2, n_folds=3)
    assert sorted(os.listdir(info["dirs"]["lesions"])) == sorted(names[i] for i in tr)


def test_missing_pair_is_an_error(tiny_data, tmp_path):
    dirs = modality_dirs(tiny_data)
    canonical = list_images(dirs["disconnectomes"])
    broken = {"lesions": str(tmp_path / "empty"), "disconnectomes": dirs["disconnectomes"]}
    os.makedirs(broken["lesions"])
    with pytest.raises(FileNotFoundError):
        make_fold_dirs(str(tmp_path / "folds"), broken, canonical, fold=0, n_folds=3)


def test_align_to_canonical_permutes_and_checks():
    canonical = ["a", "b", "c"]
    Z = np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    out = align_to_canonical(Z, ["c", "a", "b"], canonical)
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(KeyError):
        align_to_canonical(Z, ["c", "a", "zz"], canonical)
    with pytest.raises(ValueError):
        align_to_canonical(Z[:2], ["c", "a"], canonical)


def test_group_aware_fold_dirs_keep_repeats_in_training(tiny_data, tmp_path):
    pytest.importorskip("neurocausalpfn.prior.giles_replica")
    from neurocausalpfn.prior.giles_replica import group_folds  # noqa: F401  (needs main repo >= PR #33)

    dirs = modality_dirs(tiny_data)
    canonical = list_images(dirs["disconnectomes"])
    names = [os.path.basename(p) for p in canonical]
    # names[0] and names[1] are two acquisitions of one group; the rest are singles
    groups = {names[0]: ("g0", 0), names[1]: ("g0", 1)}
    for nm in names[2:]:
        groups[nm] = (f"g_{nm}", 0)
    for mode in ("giles", "strict"):
        root = str(tmp_path / mode)
        tested = []
        for fold in range(3):
            info = make_fold_dirs(root, dirs, canonical, fold, 3, groups=groups, mode=mode)
            linked = set(os.listdir(info["dirs"]["lesions"]))
            te_names = {names[i] for i in info["te_idx"]}
            assert names[1] not in te_names                       # a repeat is never tested
            if mode == "giles":
                assert names[1] in linked                          # ... and always trains
            else:
                assert (names[1] in linked) == (names[0] not in te_names)   # follows its group
            tested += sorted(te_names)
        assert sorted(tested) == sorted(set(names) - {names[1]})  # every earliest image tested once


def test_exported_fold_table_pins_the_folds_under_hashed_names(tiny_data, tmp_path):
    """The export hashes every name and writes the fold of each image; a
    listing of the hashed copies (another sort order) gives the same split."""
    pytest.importorskip("neurocausalpfn.utils.hashing")
    from neurocausalpfn.utils.hashing import hash_filename, load_salt

    from ncpfold.export_hashed import export
    from ncpfold.folds import load_groups

    dirs = modality_dirs(tiny_data)
    names = [os.path.basename(p) for p in list_images(dirs["disconnectomes"])]
    rows = [(names[0], "g0", 0), (names[1], "g0", 1)] + [(nm, f"g_{nm}", 0) for nm in names[2:]]
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text("filename,group,rank\n" + "".join(f"{n},{g},{r}\n" for n, g, r in rows))
    salt_file = tmp_path / "salt"
    salt = load_salt(str(salt_file))
    for mode in ("giles", "strict"):
        dst = tmp_path / f"export_{mode}"
        c = export(tiny_data, str(groups_csv), str(dst), salt, n_folds=3, mode=mode, scrub=(mode == "giles"))
        assert c["n"] == len(names) and c["tested"] == len(names) - 1 and c["never_tested"] == 1
        table = (dst / "fold_table.csv").read_text()
        assert "filename,group,rank,fold" in table and not any(nm in table for nm in names)
        assert not any(tok in table for tok in ("_F.", "_M.", "g0"))      # no raw name, sex or group id
        # the copies: hashed names, same count, scrubbed headers when copied
        hashed_dir = dst / os.path.basename(tiny_data) / "disconnectomes"
        copies = sorted(os.listdir(hashed_dir))
        assert copies == sorted(hash_filename(nm, salt) for nm in names)
        if mode == "giles":
            import nibabel as nib
            hdr = nib.load(str(hashed_dir / copies[0])).header
            assert hdr["descrip"].tobytes().rstrip(b"\x00") == b"" and len(hdr.extensions) == 0
        else:
            assert all(os.path.islink(hashed_dir / f) for f in copies)
        # same membership by (original) name, in the hashed listing's own order
        groups = load_groups(str(groups_csv))
        want = all_folds(len(names), 3, names=names, groups=groups, mode=mode)
        exp_groups = load_groups(str(dst / "fold_table.csv"))
        assert exp_groups.folds is not None
        got = all_folds(len(copies), 3, names=copies, groups=exp_groups, mode=mode)
        back = {hash_filename(nm, salt): nm for nm in names}
        for (tr_w, te_w), (tr_g, te_g) in zip(want, got):
            assert sorted(names[i] for i in te_w) == sorted(back[copies[i]] for i in te_g)
            assert sorted(names[i] for i in tr_w) == sorted(back[copies[i]] for i in tr_g)
        # the fold dirs of the hashed copy hold only its train side
        info = make_fold_dirs(str(dst / "folds"), modality_dirs(str(dst / os.path.basename(tiny_data))),
                              list_images(str(hashed_dir)), 1, 3, groups=exp_groups, mode=mode)
        assert sorted(os.listdir(info["dirs"]["lesions"])) == sorted(copies[i] for i in info["tr_idx"])


def test_fixed_folds_reject_inconsistent_tables():
    from ncpfold.folds import GroupTable, fixed_folds

    t = GroupTable({"a": ("g", 0), "b": ("g", 1), "c": ("h", 0)})
    t.folds = {"a": 0, "b": -1, "c": 1}
    tr, te = fixed_folds(["a", "b", "c"], t, n_folds=2, mode="strict")[0]
    assert te.tolist() == [0] and tr.tolist() == [2]                 # b follows its group out of fold 0
    tr, te = fixed_folds(["a", "b", "c"], t, n_folds=2, mode="giles")[0]
    assert tr.tolist() == [1, 2]                                      # b trains in every fold
    t.folds["b"] = 1                                                  # a repeat marked as tested
    with pytest.raises(ValueError):
        fixed_folds(["a", "b", "c"], t, n_folds=2)
    t.folds = {"a": 0, "c": 1}
    with pytest.raises(KeyError):
        fixed_folds(["a", "b", "c"], t, n_folds=2)
