# TASK.md — SkytraceRT_poc

## Current status (2026-03-14)

We have a working proof-of-concept pipeline that turns SigMF IQ captures into *controller frequency bands* suitable for downstream targeting (e.g. jammer), with outputs in **absolute Hz**.

### Data source
- GCS bucket prefix: `gs://yolo-oscilion/Yolo_Oscilion/unprocessed/capture_data_config_info/`
- Ground truth lives in each SigMF meta at:
  - `global.annotations.custom.rc_configuration`
  - `...rcs[]` with `min_frequency_mhz` / `max_frequency_mhz`

### Local subsets
- `subset20` downloaded to:
  - `/Users/omni/.openclaw/workspace/gcs_capture_data_config_info/subset20/`
- `subset100` downloaded to:
  - `/Users/omni/.openclaw/workspace/gcs_capture_data_config_info/subset100/`

Notes:
- All observed captures so far: `ci16_le`, `core:sample_rate = 245.76 Msps`, `core:sample_count = 2,097,152` (~8.53 ms).

## Pipeline (recommended)

### 1) Build dataset index (JSONL)
Script: `scripts/build_dataset_jsonl.py`
- Builds a JSONL with absolute-Hz GT band targets (`min_frequency_hz/max_frequency_hz`) + capture info.

Examples:
- `artifacts/dataset_subset20_hz.jsonl` (generated locally)
- `artifacts/dataset_subset100_hz.jsonl` (generated locally)

### 2) Build raw-IQ windows + occupancy targets
Script: `scripts/build_windows_npz.py`
- Outputs `X_i16 [N,2,L]` and `y_occ [N,F]`.

Current defaults used:
- window length = `262144` samples (~1.07 ms)
- hop = `262144` (no overlap)
- freq bins = `1024`

Large artifacts are stored outside git.

### 3) FFT/log-PSD feature frontend (fast + tiny model)
Script: `scripts/build_fft_features_npz.py`
- Converts windows NPZ into:
  - `X_feat [N,F]` float16 log-PSD features
  - `y_occ [N,F]`

### 4) Train feature model (Apple Silicon optimized)
Script: `scripts/train_feat_occ.py`
- Uses MPS if available + fp16 autocast.
- Model: `TinyFeatOccNet` (MLP over log-PSD).

Checkpoint examples (local):
- `artifacts/tiny_feat_occ_subset20_hz.pt`
- `artifacts/tiny_feat_occ_subset100_hz.pt`

### 5) Inference → controller bands
Script: `scripts/infer_feat_occ.py`
- Computes FFT/log-PSD per window, runs model, averages occupancy across windows.
- Converts occupancy to bands (absolute Hz).

## Evaluation

Scripts:
- `scripts/eval_feat_occ.py` (feature model)
- `scripts/eval_occ.py` (raw-IQ model; currently less useful)

We added simple metrics:
- mean GT coverage
- mean best-match IoU
- mean predicted band count

### Best known results (subset100)
Model: `artifacts/tiny_feat_occ_subset100_hz.pt`
Postprocess (current best for IoU with coverage constraint):
- `thr=0.50`, `hysteresis=0.10`, `smooth_radius=2`

Metrics (subset100, thr=0.5):
- mean GT coverage ~ **0.991**
- mean best-match IoU ~ **0.777**
- mean pred_n ~ **1.44**

## Post-processing improvements already implemented
File: `skytracert_poc/postprocess.py`
- Smoothing of occupancy curve
- Hysteresis thresholding
- Linear interpolation at threshold crossings for sub-bin refined edges

A sweep tool exists:
- `scripts/sweep_postprocess.py`

## Known issues / next work (tomorrow)

### A) “Tight edges” objective needs a better metric
- Current overshoot metric (`mean_best_match_overshoot`) is not useful because predicted bands can cover multiple GT bands.
- Replace with a direct edge error metric:
  - for each GT band: choose best IoU pred; compute `|dlower| + |dupper|` in Hz.
  - sweep to minimize this subject to `coverage >= target`.

### B) Reduce merged/wide bands
- `pred_n` tends to be < GT count because occupancy regions merge.
- Once edges are good, add optional splitting / peak-based segmentation.

### C) Thresholding strategy
- Add automatic per-capture threshold selection or val-tuned global threshold.

### D) Edge deployment
- Export feature model to ONNX.
- Benchmark FFT+MLP latency on Orin (TensorRT) and decide INT8 calibration.

## Repo
- GitHub: https://github.com/omni-robin/SkytraceRT_poc
