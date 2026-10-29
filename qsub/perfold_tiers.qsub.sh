#!/bin/bash -l
# Tiers 1 and 3 (ncpfold.tiers: reconstruction Dice, active dimensions, IOSS)
# of per-fold encoders, CPU only. VARLIST is a text file with one variant
# folder per line (<out-root>/<eid>/<label>/seed<S>; labels with brackets are
# read literally); every channel with a checkpoint in at least one fold gets
# one CSV in OUT_DIR, named after the folder. A CSV already there is skipped,
# so a resubmission only does what is missing.
#
#   qsub -v VARLIST=tiers_variants.txt,DATA_DIR="$HOME/Scratch/data_bcb/Full data",OUT_DIR=outputs_tiers_bcb qsub/perfold_tiers.qsub.sh
#
#$ -N ncp-tiers
#$ -l h_rt=12:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -uo pipefail
set -f                                   # variant labels such as E2[w_dice=0.1] are not globs
module load python3/3.11
source ~/venvs/neuro/bin/activate
VARLIST="${VARLIST:?set VARLIST to a file with one variant folder per line}"
DATA_DIR="${DATA_DIR:?set DATA_DIR}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
CHANNELS="${CHANNELS:-disco lesion}"
export OMP_NUM_THREADS="${NSLOTS:-4}"
mkdir -p "$OUT_DIR"
echo "perfold $(git rev-parse --short HEAD) main ${PYTHONPATH:-venv} on $(hostname)"
declare -A CKPT=([disco]=disco/vae_disconnectome.pt [lesion]=lesion/vae_lesion.pt)
fail=0
while IFS= read -r v; do
    [ -n "$v" ] || continue
    for ch in $CHANNELS; do
        n=0
        for k in 0 1 2 3 4 5 6 7 8 9; do [ -s "$v/fold$k/${CKPT[$ch]}" ] && n=$((n + 1)); done
        [ "$n" -gt 0 ] || { echo "$v $ch: no checkpoint"; continue; }
        out="$OUT_DIR/tiers_$(printf '%s' "$v" | tr '/[]=+,' '_______')_$ch.csv"
        [ -s "$out" ] && { echo "$v $ch: done before"; continue; }
        echo "== $v $ch ($n folds)"; t=$SECONDS
        if python -m ncpfold.tiers --variant-dir "$v" --data-dir "$DATA_DIR" --channel "$ch" --device cpu --out "$out"; then
            echo "   $((SECONDS - t)) s"
        else
            echo "FAILED $v $ch"; fail=1
        fi
    done
done < "$VARLIST"
exit $fail
