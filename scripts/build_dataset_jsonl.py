#!/usr/bin/env python3
"""Build dataset JSONL from SigMF meta ground-truth.

Assumes GT lives in:
  global.annotations.custom.rc_configuration.rcs[]

Writes one JSON object per capture.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing *.sigmf-meta and *.sigmf-data")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    rows: list[dict[str, Any]] = []
    for meta in sorted(in_dir.glob("*.sigmf-meta")):
        stem = meta.name.removesuffix(".sigmf-meta")
        data = in_dir / f"{stem}.sigmf-data"
        if not data.exists():
            continue

        j = load_json(meta)
        g = j.get("global") or {}
        cap0 = (j.get("captures") or [{}])[0] or {}

        rc_cfg = (g.get("annotations") or {}).get("custom", {}).get("rc_configuration") or {}
        controllers = []
        for r in rc_cfg.get("rcs") or []:
            controllers.append(
                {
                    "rc_name": r.get("rc_name"),
                    "min_frequency_hz": None if r.get("min_frequency_mhz") is None else float(r["min_frequency_mhz"]) * 1e6,
                    "max_frequency_hz": None if r.get("max_frequency_mhz") is None else float(r["max_frequency_mhz"]) * 1e6,
                    "signal_strength_dbm": r.get("signal_strength_dbm"),
                    "config_id": r.get("config_id"),
                }
            )

        rows.append(
            {
                "id": stem,
                "meta_path": str(meta.resolve()),
                "data_path": str(data.resolve()),
                "sample_rate_hz": float(g.get("core:sample_rate")) if g.get("core:sample_rate") is not None else None,
                "datatype": g.get("core:datatype"),
                "capture": {
                    "frequency_hz": cap0.get("core:frequency"),
                    "sample_count": cap0.get("core:sample_count"),
                },
                "gt": {
                    "number_of_rc": rc_cfg.get("number_of_rc"),
                    "band_type": rc_cfg.get("band_type"),
                    "controllers": controllers,
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
