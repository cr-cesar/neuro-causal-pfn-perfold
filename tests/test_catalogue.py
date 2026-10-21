import pytest

from ncpfold.catalogue import (EXCLUDED_EIDS, STUDY_EIDS, encoders_of, find_run,
                               hours_for, study_runs)
from ncpfold.plan import plan_rows


def labels(eid):
    return [s.label for s in study_runs([eid])]


def test_grids_match_the_chain():
    assert labels("E1") == ["E1"]
    assert labels("E2") == ["E2[w_dice=0.1]", "E2[w_dice=0.5]", "E2[w_dice=1.0]"]
    assert labels("E3") == ["E3[backbone=cnn]", "E3[backbone=resnet18]",
                            "E3[backbone=resnet50]", "E3[backbone=wide]"]
    assert len(labels("E4")) == 8 and "E4[100+100]" in labels("E4")
    assert labels("E6") == ["E6[fusion_mode=both]", "E6[fusion_mode=lesion]",
                            "E6[fusion_mode=disconnectome]"]
    assert set(labels("E11b")) == {"E11b[backbone=resnet18,w_dice=1.0]", "E11b[backbone=resnet18,w_dice=0.1]",
                                   "E11b[backbone=cnn,w_dice=1.0]", "E11b[backbone=cnn,w_dice=0.1]"}


def test_pins_and_pre_registered_dims():
    e1 = find_run("E1")
    assert e1.meta["w_dice"] == 0.0 and e1.meta["use_ard"] is False
    assert (e1.meta["d_lesion"], e1.meta["d_disco"]) == (50, 50)
    for eid in ("E2", "E3"):
        for sp in study_runs([eid]):
            assert (sp.meta["d_lesion"], sp.meta["d_disco"]) == (50, 50)
            assert sp.meta["use_ard"] is False
    e5 = find_run("E5")
    assert e5.meta["use_ard"] is True and e5.meta["d_lesion"] == 100
    e6 = find_run("E6", label="E6[fusion_mode=disconnectome]")
    assert e6.meta["use_ard"] is True and (e6.meta["d_lesion"], e6.meta["d_disco"]) == (100, 100)
    assert e6.meta["backbone"] == "resnet" and e6.meta["w_dice"] == 0.1


def test_find_run_by_index_and_label():
    assert find_run("E3", run=1).label == "E3[backbone=resnet18]"
    assert find_run("E3", label="E3[backbone=wide]").label == "E3[backbone=wide]"
    with pytest.raises(IndexError):
        find_run("E3", run=4)
    with pytest.raises(KeyError):
        find_run("E3", label="E3[backbone=vit]")
    for eid in EXCLUDED_EIDS:
        with pytest.raises(ValueError):
            study_runs([eid])


def test_cost_plan_counts():
    rows = plan_rows(["E2"], seeds=3, budget="chain", n_folds=10)
    assert len(rows) == 3 and all(r["tasks"] == 30 for r in rows)
    assert encoders_of(find_run("E2")) == ["vae", "vae"]           # lesion + disco
    assert encoders_of(find_run("E6", label="E6[fusion_mode=disconnectome]")) == ["vae"]
    assert hours_for(find_run("E2"), "published") < hours_for(find_run("E2"), "chain")
    whole = plan_rows(STUDY_EIDS, seeds=3, budget="chain")
    assert len(whole) == 36                                          # variants of E1..E11b
    assert sum(r["tasks"] for r in whole) == 36 * 30                 # x 3 seeds x 10 folds
    assert sum(len(r["encoders"]) for r in whole) == 62              # encoders per fold


def test_plan_only_keeps_the_named_variants_and_rejects_typos():
    import pytest
    from ncpfold.plan import plan_rows, select_rows

    rows = plan_rows(["E4", "E6", "E7b"], seeds=1, budget="published")
    kept = select_rows(rows, ["E4[25+25]", "E6:0", "E7b"])
    assert [r["label"] for r in kept] == ["E4[25+25]", "E6[fusion_mode=both]", "E7b"]
    with pytest.raises(SystemExit):
        select_rows(rows, ["E4[25+26]"])


def test_plan_emits_local_lines_for_the_sequential_runner(capsys):
    from ncpfold.plan import main

    main(["--eids", "E2", "--only", "E2:0", "--seeds", "2", "--n-folds", "3", "--emit-local",
          "--emit-local-score", "--data-dir", "export/Full data", "--out-root", "out_x",
          "--groups", "export/fold_table.csv"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    train = [ln for ln in lines if "perfold_train_local.sh" in ln]
    score = [ln for ln in lines if "ncpfold.score" in ln]
    assert len(train) == 2 * 3 and len(score) == 2
    assert train[0].startswith("JOBNAME=e2r0s0f0 EID=E2 RUN=0 SEED=0 BUDGET=chain DATA_DIR='export/Full data' "
                               "OUT=out_x GROUP_TABLE=export/fold_table.csv GROUP_MODE=giles FOLD=0 bash")
    assert "JOBNAME=score-e2r0s1" in score[1] and "--reps 'out_x/E2/E2[w_dice=0.1]/seed1/folds'" in score[1]
    assert "--groups export/fold_table.csv --test-singles" in score[1]
