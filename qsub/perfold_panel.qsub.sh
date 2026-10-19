#!/bin/bash -l
# The tabular foundation-model panel (TabPFN, TabICL) as the replica's
# classifiers on finished representations: same folds, trials and PEHE scale
# as the logistic regression / extra trees scoring, GPU, in the panel venv
# (scripts/panel_venv_myriad.sh of the main repository; torch >= 2.5).
# Weights must already be cached (warm-up on a login node, see that script).
#
#   qsub -v REPS="outputs_perfold_grp/E1/E1/seed0/folds",GROUP_TABLE=outputs_perfold/groups_public.csv,TEST_SINGLES=1,OUT=outputs_perfold_panel qsub/perfold_panel.qsub.sh
#   CLASSIFIERS="tabpfn" (default "tabpfn tabicl") picks the members.
#
#$ -N ncp-panel
#$ -l h_rt=8:0:0
#$ -l gpu=1
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source "${PANEL_VENV:-$HOME/venvs/panel}/bin/activate"
echo "node: $(hostname -s) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none) job: ${JOB_ID:-?}"

REPS="${REPS:?set REPS to one or more representation folders (globs allowed)}"
DATA_DIR="${DATA_DIR:-data/Full data}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
OUT="${OUT:-outputs_perfold_panel}"
MODALITY="${MODALITY:-receptor}"
SCENARIO="${SCENARIO:-ideal}"
CLASSIFIERS="${CLASSIFIERS:-tabpfn tabicl}"
# TabPFN: which pretrained model (v3.5 default of the package) and no browser
export TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-v3.5}" TABPFN_NO_BROWSER=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# the licence key: in a file the user alone can read, never on the command line
[ -r "$HOME/.tabpfn_token" ] && export TABPFN_TOKEN="$(cat "$HOME/.tabpfn_token")"

EXTRA=()
GT="${GROUP_TABLE:-}"
if [ -z "$GT" ] && [[ "${GROUPS:-}" == *[/.]* ]]; then GT="$GROUPS"; fi
if [ -n "$GT" ]; then EXTRA+=(--groups "$GT"); fi
if [ -n "${TEST_SINGLES:-}" ]; then EXTRA+=(--test-singles); fi
if [ -n "${ONLY_FOLDS:-}" ]; then EXTRA+=(--only-folds ${ONLY_FOLDS}); fi
if [ -n "${DEFICITS:-}" ]; then EXTRA+=(--deficits ${DEFICITS}); fi

# shellcheck disable=SC2086
python -m ncpfold.score --reps ${REPS} --data-dir "$DATA_DIR" --atlas-dir "$ATLAS_DIR" \
    --out-root "$OUT" --modality "$MODALITY" --scenario "$SCENARIO" \
    --classifiers ${CLASSIFIERS} ${EXTRA[@]+"${EXTRA[@]}"}
python -m ncpfold.leaderboard --out-root "$OUT"
