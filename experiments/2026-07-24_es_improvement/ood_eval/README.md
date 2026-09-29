# OOD ES Evaluation Pipeline

This directory contains a controlled out-of-distribution (OOD) evaluation pipeline for ES fine-tuning on math reasoning.

## Step-by-step workflow

1. **Build datasets**
   - Create train/validation split from a source dataset (`MATH` or `GSM8K`) using `ood_eval/datasets.py`.
   - Keep checkpoint selection strictly on in-domain validation split only.
2. **Train ES**
   - Run `ood_eval/run_experiment.py` with `--stage train`.
   - Checkpoints are written every fixed number of updates.
3. **Select checkpoints by ID validation only**
   - Run `--stage select` to find `best`, `final`, and optional `early` checkpoints.
4. **Evaluate selected checkpoints on ID + OOD**
   - Run `--stage eval` for pass@1, pass@k, extraction/format success, response length and compute metrics.
5. **Export aggregate artifacts**
   - `results/ood_eval_summary.csv`
   - `results/training_curves.csv`
   - plots under `plots/`

## Quick start

```bash
python ood_eval/run_experiment.py \
  --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --train_dataset math \
  --population_size 20 \
  --sigma 1e-3 \
  --alpha 5e-4 \
  --train_size 1000 \
  --val_size 500 \
  --save_every 20 \
  --stage all
```

Use this first for pipeline debugging, then scale to larger models and larger train subsets.
