#!/usr/bin/env python3
"""
Normalize Parquet schema metadata for Hugging Face datasets.

Problem: Some Parquet files have `_type: "Sequence"` in the file-level metadata
(`read_metadata(...).metadata[b"huggingface"]`) but still contain `_type: "List"`
in the schema-level metadata (`read_schema(...).metadata[b"huggingface"]`).

`datasets`' parquet builder reads from the schema-level metadata, which can lead to
errors like: `ValueError: Feature type 'List' not found`.

Fix: For every `.parquet` file under `--root`, read the file-level `huggingface` JSON
and write it into the schema metadata. This preserves all other schema metadata keys.

Usage:
  python scripts/normalize_parquet_hf_schema_meta.py --root /path/to/dataset [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def normalize_file(path: Path, dry_run: bool = True) -> tuple[bool, str]:
    """Return (changed, message)."""
    md = pq.read_metadata(str(path))
    schema = pq.read_schema(str(path))

    file_meta = md.metadata or {}
    schema_meta = dict(schema.metadata or {})

    hf_file = file_meta.get(b"huggingface")
    hf_schema = schema_meta.get(b"huggingface")

    if not hf_file and not hf_schema:
        return False, "no hf metadata"

    # Prefer file-level HF JSON when present (typically the authoritative one)
    source = hf_file or hf_schema

    try:
        src_json = json.loads(source.decode("utf-8"))
    except Exception as e:
        return False, f"invalid hf json: {e}"

    # If schema already matches file-level, no change
    if hf_schema == source:
        return False, "schema up-to-date"

    # Write new schema metadata with the HF JSON from file-level
    if dry_run:
        return True, "would update schema hf metadata"

    table = pq.read_table(str(path))
    existing = dict(table.schema.metadata or {})
    existing[b"huggingface"] = json.dumps(src_json, ensure_ascii=False).encode("utf-8")
    new_table = table.replace_schema_metadata(existing)
    # Overwrite in-place
    pq.write_table(new_table, str(path))
    return True, "updated schema hf metadata"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="Root directory containing parquet files")
    ap.add_argument("--dry-run", action="store_true", help="Only print actions; don't modify files")
    args = ap.parse_args()

    root: Path = args.root
    dry_run: bool = args.dry_run

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    total = 0
    changed = 0
    skipped = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".parquet"):
                continue
            total += 1
            path = Path(dirpath) / fname
            ch, msg = normalize_file(path, dry_run=dry_run)
            if ch:
                changed += 1
            else:
                skipped += 1
            print(f"[{('DRY' if dry_run else 'APPLY')}] {path}: {msg}")

    print(f"\nSummary: total={total}, changed={changed}, skipped={skipped}")


if __name__ == "__main__":
    main()

