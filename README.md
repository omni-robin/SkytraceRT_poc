# SkytraceRT_poc

Proof-of-concept: **raw-IQ → controller-band occupancy → (lower/upper/center) bands** for downstream targeting (e.g. jamming).

## Goal
Given SigMF IQ captures with ground-truth controller bands in:

`global.annotations.custom.rc_configuration.rcs[]`

train a small, fast model that can run on edge devices (e.g. Nvidia Orin) and output:

- estimated controller count (derived)
- per-controller frequency bands: `lower_hz`, `upper_hz`, `center_hz`

Absolute Hz output is the default.

## Approach (recommended)
Model predicts a **1D frequency occupancy mask** over the capture span.
A lightweight post-process converts the mask into N bands.

Why this formulation:
- variable number of controllers handled naturally
- stable + tunable post-processing without retraining
- easy to deploy (ONNX → TensorRT)

## Quickstart (Apple Silicon)

Create venv + install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy torch
```

Build dataset index (JSONL):

```bash
python scripts/build_dataset_jsonl.py \
  --in-dir /path/to/sigmf_pairs \
  --out artifacts/dataset.jsonl
```

Build raw-IQ windows + occupancy targets:

```bash
python scripts/build_windows_npz.py \
  --dataset-jsonl artifacts/dataset.jsonl \
  --out artifacts/windows.npz \
  --win-len 262144 --win-hop 262144 \
  --freq-bins 1024
```

Train (uses MPS if available):

```bash
python scripts/train_occ.py \
  --npz artifacts/windows.npz \
  --out artifacts/tiny_occ.pt \
  --epochs 15 --batch-size 32
```

Infer one capture → absolute-Hz controller bands:

```bash
python scripts/infer_occ.py \
  --ckpt artifacts/tiny_occ.pt \
  --meta /path/to/file.sigmf-meta \
  --data /path/to/file.sigmf-data
```

## Repo layout
- `scripts/`  dataset preparation + QC + training/inference
- `skytracert_poc/`  python package (model + dataset + postprocess)

## Notes
- Training is optimized for Apple Silicon by defaulting to the MPS backend and using fp16 autocast.
