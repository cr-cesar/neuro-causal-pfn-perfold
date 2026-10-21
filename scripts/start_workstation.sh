#!/bin/bash
# On the workstation: unpack the code next to the export, install the per-fold
# package into the Python environment that already holds torch (VENV,
# default ~/env) and start a job list in the background, so that it keeps
# running after the remote session closes.
#
#   tar -xzf export_bcb.tgz && bash export_bcb/start_workstation.sh            # pilot list
#   bash export_bcb/start_workstation.sh jobs/study_chain_train.txt            # the study
#   RUNNERS=3 bash export_bcb/start_workstation.sh jobs/study_chain_train.txt  # three runners on one GPU
#
# Layout after this script: ~/neuro-causal-pfn (main), ~/neuro-causal-pfn-perfold
# (study), ~/<export> next to them. Progress: tail -f ~/neuro-causal-pfn-perfold/run_*.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
JOBS="${1:-jobs/pilot_e2_e9a_fold0.txt}"
RUNNERS="${RUNNERS:-1}"
export VENV="${VENV:-$HOME/env}"
[ -x "$VENV/bin/python" ] || { echo "no Python at $VENV/bin/python (set VENV)"; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for c in "$HERE"/code/*.tgz; do
    [ -s "$c" ] && tar -xzf "$c"                      # overwrites the code, keeps outputs/ and logs/
done
[ -d neuro-causal-pfn ] && [ -d neuro-causal-pfn-perfold ] || { echo "code archives missing from $HERE/code"; exit 2; }
PIP() { if [ -x "$VENV/bin/pip" ]; then "$VENV/bin/pip" "$@"; else uv pip "$1" --python "$VENV/bin/python" "${@:2}"; fi; }
"$VENV/bin/python" -c "import neurocausalpfn" 2>/dev/null || PIP install -q --no-build-isolation --no-deps -e neuro-causal-pfn
PIP install -q --no-build-isolation --no-deps -e neuro-causal-pfn-perfold
"$VENV/bin/python" -c "import torch, nibabel, ncpfold; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| ncpfold ok')"
mkdir -p neuro-causal-pfn-perfold/data
[ -e neuro-causal-pfn-perfold/data/atlases ] || cp -r "$HERE/atlases" neuro-causal-pfn-perfold/data/
cd neuro-causal-pfn-perfold
mkdir -p jobs && cp -n "$HERE"/jobs/*.txt jobs/
[ -s "$JOBS" ] || { echo "no jobs file $JOBS"; exit 2; }
n=$(grep -c -v -E '^\s*(#|$)' "$JOBS")
stamp="$(date +%Y%m%d_%H%M)"
if [ "$RUNNERS" -gt 1 ]; then
    rm -f "jobs/part_${stamp}_"*; grep -v -E '^\s*(#|$)' "$JOBS" | split -n "l/$RUNNERS" -d - "jobs/part_${stamp}_"
    for part in jobs/part_${stamp}_*; do
        nohup env WORKDIR="$PWD" bash ../neuro-causal-pfn/scripts/run_jobs.sh "$part" > "run_${stamp}_$(basename "$part").log" 2>&1 &
    done
    echo "started $n jobs from $JOBS on $RUNNERS runners; progress: tail -f $PWD/run_${stamp}_*.log"
else
    nohup env WORKDIR="$PWD" bash ../neuro-causal-pfn/scripts/run_jobs.sh "$JOBS" > "run_${stamp}.log" 2>&1 &
    echo "started $n jobs from $JOBS in the background (pid $!); progress: tail -f $PWD/run_${stamp}.log"
fi
echo "results: $PWD/outputs_perfold_ws/<eid>/<label>/seed<s>/folds/fold<k>.npz; scores after jobs/study_*_score.txt"
