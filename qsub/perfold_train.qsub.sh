#!/bin/bash -l
# One variant x one seed, all folds as an SGE array (task K = fold K-1).
#
#   qsub -t 1-10 -v EID=E5,RUN=0,SEED=0 qsub/perfold_train.qsub.sh
#   qsub -t 1-10 -v EID=E3,RUN=2,SEED=0,BUDGET=published qsub/perfold_train.qsub.sh
#   python -m ncpfold.plan --eids E5 --seeds 1 --emit-qsub > submit.sh && bash submit.sh
#
# GROUPS=<filename,group,rank csv> (+ GROUP_MODE=giles|strict) switches to
# group-aware folds; use a separate OUT for that protocol, e.g. OUT=outputs_perfold_grp.
# RUN is the variant index printed by `python -m ncpfold.plan --eids <EID>`
# (labels contain commas, which qsub -v cannot carry). Re-running a finished
# fold is a no-op: the fold npz is skipped unless FORCE=1.
#
# Wall clock: a chain-budget fold trains up to two 200-epoch VAEs (~2.5 h on
# A100, ~5 h for resnet50) plus the encoding pass; 8 h covers every variant on
# A100. On V100 nodes chain-budget resnet50 folds need 12 h: raise h_rt or pin
# A100 (-ac allow=L).
#
#$ -N ncp-fold
#$ -l h_rt=8:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=6G
#$ -l tmpfs=20G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

EID="${EID:?set EID, e.g. -v EID=E5,RUN=0,SEED=0}"
RUN="${RUN:-0}"
SEED="${SEED:-0}"
BUDGET="${BUDGET:-chain}"
DATA_DIR="${DATA_DIR:-data/Full data}"
OUT="${OUT:-outputs_perfold}"
TASK="${TASK:-disconnectome}"
FOLD=$((SGE_TASK_ID - 1))

EXTRA=()
if [ -n "${FORCE:-}" ]; then EXTRA+=(--force); fi
if [ -n "${GROUPS:-}" ]; then EXTRA+=(--groups "$GROUPS" --group-mode "${GROUP_MODE:-giles}"); fi

python -m ncpfold.perfold --eid "$EID" --run "$RUN" --seed "$SEED" --fold "$FOLD" \
    --data-dir "$DATA_DIR" --out-root "$OUT" --task "$TASK" --budget "$BUDGET" \
    ${EXTRA[@]+"${EXTRA[@]}"}
