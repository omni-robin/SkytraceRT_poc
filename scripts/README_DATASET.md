# Dataset notes

Ground truth is expected inside each SigMF meta file:

- `global.annotations.custom.rc_configuration`
  - `number_of_rc`
  - `band_type`
  - `rcs[]`: each has at least `min_frequency_mhz` and `max_frequency_mhz`

The scripts in this repo build a JSONL index with absolute-Hz band targets.

## Build JSONL index

```bash
python3 scripts/build_dataset_jsonl.py \
  --in-dir /path/to/subset \
  --out artifacts/dataset.jsonl
```
