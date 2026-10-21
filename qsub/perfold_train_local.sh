#!/bin/bash
# One fold of one variant on a machine without a scheduler (a workstation
# with one GPU). Same variables as qsub/perfold_train.qsub.sh plus FOLD
# (0-based) and VENV (the Python environment, default ~/env); lines for the
# whole study come from `python -m ncpfold.plan --emit-local` and run with
# the main repository's scripts/run_jobs.sh:
#
#   EID=E2 RUN=0 SEED=0 FOLD=3 DATA_DIR=export/Full\ data OUT=outputs_perfold_x \
#     GROUP_TABLE=export/fold_table.csv bash qsub/perfold_train_local.sh
#
# A finished fold (its npz exists) is a no-op unless FORCE=1.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-$HOME/env}"
[ -x "$VENV/bin/python" ] || { echo "no Python at $VENV/bin/python (set VENV)"; exit 2; }
export PATH="$VENV/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
echo "node: $(hostname -s) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"

EID="${EID:?set EID, e.g. EID=E5 RUN=0 SEED=0 FOLD=0}"
FOLD="${FOLD:?set FOLD (0-based)}"
EXTRA=()
[ -n "${FORCE:-}" ] && EXTRA+=(--force)
if [ -n "${GROUP_TABLE:-}" ]; then
    [ -f "$GROUP_TABLE" ] || { echo "group table not found: $GROUP_TABLE"; exit 1; }
    EXTRA+=(--groups "$GROUP_TABLE" --group-mode "${GROUP_MODE:-giles}")
fi
echo "fold $FOLD | group table: ${GROUP_TABLE:-none (image-level folds)} | mode ${GROUP_MODE:-giles}"
python -m ncpfold.perfold --eid "$EID" --run "${RUN:-0}" --seed "${SEED:-0}" --fold "$FOLD" \
    --data-dir "${DATA_DIR:-data/Full data}" --out-root "${OUT:-outputs_perfold}" \
    --task "${TASK:-disconnectome}" --budget "${BUDGET:-chain}" ${EXTRA[@]+"${EXTRA[@]}"}
