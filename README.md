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

## Repo layout
- `scripts/`  dataset preparation + QC
- `skytracert_poc/`  python package (model + training code)

## Status
- scaffolding
