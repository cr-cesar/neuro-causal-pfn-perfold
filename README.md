# neuro-causal-pfn-perfold

Per-fold replication of the Neuro-Causal-PFN Phase-1 encoder chain under the
Giles et al. (2025) protocol: every representation is fitted inside each
cross-validation fold, on the fold's training side only, and used frozen to
encode both sides. The Phase-1 chain trained each encoder once on the whole
cohort before the split (a documented deviation); this study removes it for
E1 to E11 and scores everything on the same virtual-trial replica.

Nothing is re-implemented. The trainers, the Table-9 registry, the variant
builder and the replica scorer are imported from the main package
[`neurocausalpfn`](https://github.com/cr-cesar/neuro-causal-pfn); this
repository only adds the fold protocol around them.

## Protocol

1. **Folds.** `KFold(10, shuffle=True, random_state=0)` over the sorted file
   listing of the task modality (disconnectomes by default), the exact folds
   of the replica. They are fixed before any model is fitted.
2. **Training.** For fold K, a folder of symlinks to the train-side files is
   built for both modalities and every trainer's data root points at it. The
   trainer's own `full_config()` and epochs are used (`--budget chain`);
   `--budget published` caps the VAEs at 32 epochs, the paper's budget.
3. **Encoding.** The frozen encoder (best-validation checkpoint for the VAE
   family, final checkpoint for the others) encodes the full listing; the
   rows are split into `Ztr` / `Zte` by the fold indices.
4. **Scoring.** `evaluate_representation` of the replica receives a per-fold
   lookup, so each test fold is scored with latents its encoder never saw.
   Headline: `pehe_paper_mean` (all participants, tau in {-1, 0, +1}).

Seeds: study seed `s` trains fold K with seed `1000*s + K`.

## Install (Myriad)

```bash
module load python3/3.11 && source ~/venvs/neuro/bin/activate   # has neurocausalpfn -e
git clone <this repo> ~/Scratch/neuro-causal-pfn-perfold
cd ~/Scratch/neuro-causal-pfn-perfold
pip install -e .
ln -s ~/Scratch/neuro-causal-pfn/data data          # lesions, disconnectomes, atlases
python -m pytest tests -q                            # CPU, ~1-2 min (tiny synthetic cohort)
```

## Run

```bash
python -m ncpfold.plan --eids E1 E5 --seeds 1                 # variants, indices, GPU-hours
python -m ncpfold.plan --eids E1 E5 --seeds 1 --emit-qsub > submit.sh && bash submit.sh
# ... one array job (10 folds) per variant and seed; when they finish:
qsub -v REPS="outputs_perfold/E1/E1/seed0/folds outputs_perfold/E5/E5/seed0/folds",WITH_VOLUME=1 \
     qsub/perfold_score.qsub.sh
python -m ncpfold.leaderboard --out-root outputs_perfold
```

A single fold by hand:

```bash
python -m ncpfold.perfold --eid E3 --run 1 --seed 0 --fold 4 --data-dir "data/Full data"
```

Layout: `outputs_perfold/<eid>/<label>/seed<S>/fold<K>/` (checkpoints),
`.../seed<S>/folds/fold<K>.npz` (latents), `outputs_perfold/replica/...`
(scores), `outputs_perfold/folds/fold<K>/` (symlink folders, shared by all
variants), `outputs_perfold/labels/` (cached anatomical labels).

## Scope and cost

| Experiments | Variants | Encoders per fold | 3 seeds x 10 folds | chain budget | published budget |
|---|---|---|---|---|---|
| E1, E2, E3, E4, E5, E5b, E6, E7a, E7b, E9a-c, E10a-c, E11a, E11b | 36 | 62 | 1,080 GPU array tasks | ~2,500 GPU-h | ~400 GPU-h |

Unlike the chain, a variant here never reuses another variant's encoder
(E6's single channels and E11b's disconnectomes are retrained per variant),
which is what a clean per-fold protocol requires and why the encoder count is
higher than the chain's.

`python -m ncpfold.plan` prints the exact figures. Per-encoder times are the
Phase-1 measurements for the VAEs (1.2 h per 200-epoch VAE on A100, 2.3 h for
resnet50); the non-VAE trainers are estimates until their first fold is timed
(`catalogue.HOURS_PER_ENCODER`). Excluded: E0 (the scorer computes the volume
and per-fold NMF-50 references itself: `--with-volume`, `--with-nmf`), E8 and
E5c (need the clinical CSV), E12 (Phase 2).

## What is different from the Phase-1 chain, on purpose

- Every variant sees the final pinned context (backbone resnet, lambda_Dice
  0.1, 100+100 dims, ARD, disconnectome channel). E4 therefore trains its
  lesion encoders with lambda_Dice 0.1, not the 0.5 of its historical run.
  E2 and E3 stay at the pre-registered 50+50, as in the registry.
- Reference points on the same folds: the Phase-1 plan-B VAE-50 (per fold,
  published budget) scored 0.294; the chain's whole-cohort E1 scored 0.320
  and E5 0.317. Compare like with like: this study's E1 (chain budget) is the
  new baseline for this study's E2 to E11.

## Status

- Adapters for all six trainer kinds (`ncpfold/encoders.py`); the VAE family,
  early fusion and DMVAE are exercised end to end by the tests on a tiny
  synthetic cohort. Contrastive, MAE and DSCM adapters follow the same
  pattern and are validated by their first real fold.
- Tests run on CPU without the real data.
