# PaliGemma 2 Reproduction Toolkit

This directory extends the repository with scripts and modules to reproduce and extend the **PaliGemma 2** training setup where a SigLIP-So400m vision encoder is paired with a Gemma 2 text decoder. The focus is on three-stage curriculum training, high-frequency checkpointing, and developmental-interpretability metrics.

## Released assets

All public checkpoints and documentation are discoverable on the Hugging Face Hub via Google DeepMind releases. The helper script `scripts/verify_assets.py` checks availability of:

- PaliGemma 2 weights (`3B`, `10B`, `28B`) at input resolutions `224`, `448`, and `896`.
- The SigLIP-So400m vision tower (`google/siglip-so400m-patch14-384`).
- Gemma 2 instruction-tuned decoders (`google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-2-27b-it`).
- Public documentation/model cards (tech report and per-checkpoint cards).

```bash
python scripts/verify_assets.py
```

To download weights and optionally verify checksums/model cards:

```bash
python scripts/download_assets.py --key 3b-224 --output-dir weights --dump-card
```

SHA-256 hashes can be attached to `paligemma2.modeling.PALIGEMMA2_ASSETS` once publicly released. The downloader will warn if hashes are absent.

## Environment setup

```bash
pip install -r requirements.txt  # provide torch>=2.3, transformers>=4.41, huggingface_hub, pandas, pyarrow, tensorboard
```

Optional dependencies:

- `devinterp` for Log-Likelihood Curvature (LLC) estimation hooks.
- `flash-attn` / PyTorch nightly if you plan to train the larger checkpoints.

## Repository layout additions

```
paligemma2/
  config.py               # stage configs and asset metadata
  modeling.py             # SigLIP + Gemma fusion module + asset catalog
  data/registry.py        # task registry with stubbed loaders (replace with real datasets)
  training/pipeline.py    # three-stage trainer with high-frequency checkpointing
metrics/
  training_dev_metrics.py # developmental metrics streamer
scripts/
  download_assets.py      # weight/model-card downloader with hash verification
  verify_assets.py        # probe availability of public assets
  train.py                # curriculum trainer entry point
  eval.py                 # checkpoint evaluation across registered tasks
  export_activations.py   # post-hoc activation exporter
  checkpoint_indexer.py   # summarise CHECKPOINT_MANIFEST.parquet
```

Artifacts written during training:

- `CHECKPOINT_MANIFEST.parquet` – index of saved checkpoints with parameter descriptors.
- `METRICS/train_timeseries.parquet` – streaming timeseries for loss/geometry/LLC metrics.
- Optional TensorBoard logs if `MetricConfig.tb_dir` is provided.

## Three-stage training recipe

The default schedule mirrors the public recipe:

1. **Stage 1** – 224px, ~1B examples, joint multimodal mix.
2. **Stage 2** – split into 50M 448px and 10M 896px steps with OCR upweighting.
3. **Stage 3** – per-task fine-tunes at 896px reusing Stage 2 hyperparameters.

Hyperparameters live in `StageSchedule.default_schedule()` and are easily overridden.

Launch training (smoke-test with short stage lengths):

```bash
python scripts/train.py --model-size 3b --precision bf16 --max-steps 10 --save-every 5 --log-dir runs/3b_smoke
```

By default the data registry uses synthetic placeholders – replace `paligemma2/data/registry.py` loaders with real datasets or plug in your own modules through the registry API.

## Evaluation and post-hoc tooling

Evaluate a checkpoint across registered tasks:

```bash
python scripts/eval.py --checkpoint checkpoints/ckpt_step00000200.pt --model-size 3b --resolution 448
```

Export activation snapshots for interpretability probes:

```bash
python scripts/export_activations.py --checkpoint checkpoints/ckpt_step00000200.pt \
  --layers text_model.model.layers.0 --out activations/layer0.parquet --batches 4
```

Build a condensed manifest summary:

```bash
python scripts/checkpoint_indexer.py --log-dir runs/3b_smoke --out runs/3b_smoke/manifest_summary.json
```

## Developmental metrics

`metrics/training_dev_metrics.py` streams:

- Core training stats (loss, LR, batch size, gradient/weight norms).
- Grad-to-weight ratios and optimiser momentum.
- Hutchinson trace estimates for Hessian curvature (requires registering a closure each step, handled by the trainer).
- Optional LLC estimates via DevInterp SGLD sampling (guarded when the package is unavailable).
- Parameter-space descriptors (layer norms, singular value proxies, sparsity, cosine drift).
- Plugin module hook for custom metrics.

Outputs are stored as Parquet and TensorBoard scalars to enable downstream developmental-interpretability analysis.

## Data mixture placeholders

Because the original pretraining mixture is not public, the registry wires in synthetic datasets that emulate the expected tensor shapes. Replace the loader functions with your task-specific datasets (captioning, OCR-heavy, grounded captioning, detection/segmentation) and adjust the per-stage `task_weights` in `paligemma2/config.py`.

## Testing

Run unit tests to validate metric streaming and manifest handling:

```bash
pytest tests
```

## Caveats

- Full pretraining datasets and certain fine-tuning corpora are not publicly released; stubs are provided instead.
- LLC estimation can be computationally expensive and may require tuning sampler hyperparameters for stability.
- High-resolution (896px) stages require GPUs with large memory footprints; consider gradient checkpointing/FSDP.

