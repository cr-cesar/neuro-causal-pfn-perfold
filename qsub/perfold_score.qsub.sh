#!/bin/bash -l
# Replica scoring of finished representations (CPU only). REPS is a
# space-separated list of representation folders (fold*.npz + meta.json);
# globs are expanded, so a whole experiment can go in one job:
#
#   qsub -v REPS="outputs_perfold/E5/*/seed?/folds" qsub/perfold_score.qsub.sh
#   qsub -hold_jid <train job id> -v REPS="outputs_perfold/E1/E1/seed0/folds",WITH_VOLUME=1 ...
#
# Each representation takes about the time of one Phase-1 replica run
# (16 deficits x 10 folds x 4 estimators); budget ~1 h each on 4 cores.
#
#$ -N ncp-score
#$ -l h_rt=12:0:0
#$ -pe smp 4
#$ -l mem=6G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

REPS="${REPS:?set REPS to one or more representation folders (globs allowed)}"
DATA_DIR="${DATA_DIR:-data/Full data}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
OUT="${OUT:-outputs_perfold}"
MODALITY="${MODALITY:-receptor}"
SCENARIO="${SCENARIO:-ideal}"

EXTRA=()
if [ -n "${WITH_VOLUME:-}" ]; then EXTRA+=(--with-volume); fi
if [ -n "${WITH_NMF:-}" ]; then EXTRA+=(--with-nmf); fi
if [ -n "${GROUPS:-}" ]; then EXTRA+=(--groups "$GROUPS"); fi

# shellcheck disable=SC2086
python -m ncpfold.score --reps ${REPS} --data-dir "$DATA_DIR" --atlas-dir "$ATLAS_DIR" \
    --out-root "$OUT" --modality "$MODALITY" --scenario "$SCENARIO" ${EXTRA[@]+"${EXTRA[@]}"}
python -m ncpfold.leaderboard --out-root "$OUT"
