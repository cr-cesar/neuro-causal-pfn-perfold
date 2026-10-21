#!/bin/bash -l
# Build the package a workstation needs to run the per-fold study: the cohort
# exported under hashed names with the folds pinned (ncpfold.export_hashed),
# the job lists for the sequential runner, and the code of both repositories.
# A short CPU job (the images are rewritten with scrubbed headers).
#
#   # on a login node first (seconds): the code archives of the development branch
#   for r in neuro-causal-pfn neuro-causal-pfn-perfold; do (cd ~/Scratch/$r && git fetch -q origin claude/phase-1-vae-encoder-f02ugc \
#       && git archive --format=tar.gz --prefix=$r/ -o ~/Scratch/code_$r.tgz FETCH_HEAD); done
#   qsub -v DATA_DIR="$HOME/Scratch/data_bcb/Full data",GROUP_TABLE=outputs_perfold/groups_public.csv,DST=$HOME/Scratch/export_bcb \
#        qsub/export_hashed.qsub.sh
#
# Result: $DST.tgz holding export_bcb/{Full data, fold_table.csv, jobs/*.txt,
# code/*.tgz, start_workstation.sh}. Prints counts and sizes only.
#
#$ -N ncp-export
#$ -l h_rt=2:0:0
#$ -pe smp 2
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail
module load python3/3.11
source ~/venvs/neuro/bin/activate
export HASH_SALT_FILE="${HASH_SALT_FILE:-$HOME/.cache_salt}"

DATA_DIR="${DATA_DIR:?set DATA_DIR (tier folder with lesions/ and disconnectomes/)}"
GROUP_TABLE="${GROUP_TABLE:?set GROUP_TABLE (filename,group,rank csv)}"
DST="${DST:?set DST (export folder; <DST>.tgz is written next to it)}"
CODE="${CODE:-$HOME/Scratch/code_neuro-causal-pfn.tgz $HOME/Scratch/code_neuro-causal-pfn-perfold.tgz}"
GROUP_MODE="${GROUP_MODE:-giles}"
TIER="$(basename "$DATA_DIR")"
OUT_ROOT="${OUT_ROOT:-outputs_perfold_ws}"
name="$(basename "$DST")"

python -m ncpfold.export_hashed --data-dir "$DATA_DIR" --groups "$GROUP_TABLE" --dst "$DST" --group-mode "$GROUP_MODE"

# job lists, with paths relative to the workstation's per-fold checkout
# (the export is unpacked next to the two repositories: ../<name>)
mkdir -p "$DST/jobs"
D="../$name/$TIER"; G="../$name/fold_table.csv"
P=(python -m ncpfold.plan --data-dir "$D" --groups "$G" --group-mode "$GROUP_MODE" --out-root "$OUT_ROOT")
"${P[@]}" --budget chain --emit-local --only E2:0 E9a --seeds 1 --n-folds 10 \
    | awk 'NR==1 || /FOLD=0 /' > "$DST/jobs/pilot_e2_e9a_fold0.txt"
"${P[@]}" --budget chain --emit-local > "$DST/jobs/study_chain_train.txt"
"${P[@]}" --budget chain --emit-local-score > "$DST/jobs/study_chain_score.txt"
"${P[@]}" --budget published --emit-local > "$DST/jobs/study_published_train.txt"
"${P[@]}" --budget published --emit-local-score > "$DST/jobs/study_published_score.txt"
mkdir -p "$DST/code"
for c in $CODE; do [ -s "$c" ] && cp "$c" "$DST/code/" || echo "warning: no code archive $c"; done
cp scripts/start_workstation.sh "$DST/"
cp -rL data/atlases "$DST/"          # data -> the main repository's data folder; the atlases are public

# nothing identifiable: every image name is 16 hex digits, the table has no raw name
bad=$(find "$DST/$TIER" -name '*.nii*' | xargs -n1 basename | grep -v -c -E '^[0-9a-f]{16}\.nii(\.gz)?$' || true)
[ "$bad" -eq 0 ] || { echo "export check: $bad files with raw names, stopping"; exit 2; }
echo "export check: clean; jobs: $(wc -l < "$DST/jobs/study_chain_train.txt") train lines, $(wc -l < "$DST/jobs/study_chain_score.txt") score lines"
tar -C "$(dirname "$DST")" -czf "$DST.tgz" "$name"
du -sh "$DST" "$DST.tgz"
