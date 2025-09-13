#!/usr/bin/env python3
"""
One-pass fixer for Hugging Face parquet metadata inconsistencies.

Combines the behavior of:
 - scripts/normalize_parquet_hf_schema_meta.py
 - scripts/convert_parquet_hf_list_to_sequence.py

Actions performed per .parquet file:
 1) Reads both file-level and schema-level `huggingface` JSON metadata.
 2) Picks an authoritative source (by default: prefer file-level when present).
 3) Converts any feature entries with `_type: "List"` to `_type: "Sequence"`.
 4) Writes the resulting JSON into the Parquet schema metadata and rewrites the file
    (or mirrors to `--out`) so file-level and schema-level metadata are consistent.

Safe defaults:
 - Dry-run by default; use `--apply` to write changes.
 - Can mirror to a separate `--out` directory or modify in-place.

Usage examples:
  Dry-run, in-place analysis
    python scripts/fix_parquet_hf_metadata.py --root /path/to/dataset

  Apply in-place
    python scripts/fix_parquet_hf_metadata.py --root /path/to/dataset --apply

  Mirror to a separate output directory
    python scripts/fix_parquet_hf_metadata.py --root /path/to/src --out /path/to/dst --apply

Notes on --out:
  If you pass a directory like "--out dataset", outputs are nested under
  "dataset/<basename_of_root>/..." automatically, e.g. root=/data/foo/bar -> dataset/bar/...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pyarrow.parquet as pq


@dataclass
class HFMeta:
    source: str  # "file" | "schema" | "none"
    json: Optional[dict]
    raw_file: Optional[bytes]
    raw_schema: Optional[bytes]


def iter_all_paths(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        yield d, True
        for fname in filenames:
            yield d / fname, False


def read_hf_metadata(pq_path: Path) -> HFMeta:
    md = pq.read_metadata(str(pq_path))
    schema = pq.read_schema(str(pq_path))
    file_meta = md.metadata or {}
    schema_meta = schema.metadata or {}
    raw_file = file_meta.get(b"huggingface")
    raw_schema = schema_meta.get(b"huggingface")

    # Choose a source without decoding first
    src_bytes: Optional[bytes]
    src = "none"
    if raw_file:
        src_bytes = raw_file
        src = "file"
    elif raw_schema:
        src_bytes = raw_schema
        src = "schema"
    else:
        return HFMeta(source="none", json=None, raw_file=None, raw_schema=None)

    try:
        data = json.loads(src_bytes.decode("utf-8"))
    except Exception:
        data = None

    return HFMeta(source=src, json=data, raw_file=raw_file, raw_schema=raw_schema)


def convert_list_to_sequence(hf: dict) -> Tuple[dict, int, list]:
    if not isinstance(hf, dict):
        return hf, 0
    info = hf.get("info") or {}
    feats = info.get("features")
    if not isinstance(feats, dict):
        return hf, 0
    changes = 0
    changed_names = []
    for name, spec in feats.items():
        if isinstance(spec, dict) and spec.get("_type") == "List":
            spec["_type"] = "Sequence"
            changes += 1
            changed_names.append(name)
    if changes:
        hf.setdefault("info", {}).setdefault("features", feats)
    return hf, changes, changed_names


def write_with_hf_metadata(src: Path, dest: Optional[Path], hf_json: dict):
    table = pq.read_table(str(src))
    schema = table.schema
    existing = dict(schema.metadata or {})
    existing[b"huggingface"] = json.dumps(hf_json, ensure_ascii=False).encode("utf-8")
    new_table = table.replace_schema_metadata(existing)
    if dest is None or dest == src:
        # Atomic in-place rewrite
        with tempfile.NamedTemporaryFile(dir=str(src.parent), suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            pq.write_table(new_table, str(tmp_path))
            os.replace(str(tmp_path), str(src))
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            raise
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="Root directory containing the dataset")
    ap.add_argument("--out", type=Path, default=None, help="Mirror output directory (optional)")
    ap.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of parquet files processed")
    ap.add_argument(
        "--prefer",
        choices=["file", "schema", "auto"],
        default="auto",
        help="Which metadata to prefer when both exist (default: auto=file then schema)",
    )
    args = ap.parse_args()

    root: Path = args.root
    out_root_base: Optional[Path] = args.out.resolve() if args.out else None
    apply: bool = args.apply
    limit: Optional[int] = args.limit
    prefer: str = args.prefer

    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        raise SystemExit(2)

    seen_parquet = 0
    affected_parquet = 0
    changed_entries = 0
    copied_other = 0
    total_parquet = 0

    root_resolved = root.resolve()
    # If out is provided, auto-nest under <out>/<root_basename>
    out_root = (out_root_base / root_resolved.name) if out_root_base else None

    for path, is_dir in iter_all_paths(root_resolved):
        rel = path.relative_to(root_resolved)
        if is_dir:
            if out_root:
                dest_dir = out_root / rel
                if apply:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"[DRY-RUN] mkdir {dest_dir}")
            continue

        if path.suffix != ".parquet":
            # passthrough copy for non-parquet when mirroring
            if out_root:
                dest = out_root / rel
                if apply:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                print(f"[{('APPLIED' if apply else 'DRY-RUN')}] copy {path} -> {dest}")
                copied_other += 1
            continue

        total_parquet += 1
        if limit is not None and seen_parquet >= limit:
            continue
        seen_parquet += 1

        hf_meta = read_hf_metadata(path)
        if hf_meta.source == "none" or hf_meta.json is None:
            # No usable HF metadata; just mirror/copy-through
            if out_root:
                dest = out_root / rel
                if apply:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                print(f"[{('APPLIED' if apply else 'DRY-RUN')}] copy {path} -> {dest} (no hf meta)")
            else:
                print(f"[SKIP] {path} (no hf meta)")
            continue

        # Choose authoritative source if caller forces a preference
        chosen_json = hf_meta.json
        if prefer == "schema" and hf_meta.raw_schema:
            try:
                chosen_json = json.loads(hf_meta.raw_schema.decode("utf-8"))
            except Exception:
                pass
        elif prefer == "file" and hf_meta.raw_file:
            try:
                chosen_json = json.loads(hf_meta.raw_file.decode("utf-8"))
            except Exception:
                pass

        updated_json, n_changes, changed_names = convert_list_to_sequence(chosen_json)

        # Determine if rewrite is needed by comparing the target JSON to existing raw bytes
        target_bytes = json.dumps(updated_json, ensure_ascii=False).encode("utf-8")
        schema_same = hf_meta.raw_schema == target_bytes
        file_same = hf_meta.raw_file == target_bytes if hf_meta.raw_file is not None else False
        needs_write = not (schema_same and file_same)

        if not needs_write and out_root is None:
            # Nothing to do in-place
            print(f"[OK] {path} (no change)")
            continue

        affected_parquet += 1 if needs_write else 0
        changed_entries += n_changes

        dest = (out_root / rel) if out_root else None
        if apply:
            write_with_hf_metadata(path, dest, updated_json)
            label = "write" if dest else "update"
            extra = f" | names: {', '.join(changed_names)}" if n_changes else ""
            print(f"[APPLIED] {label} {dest or path} | entries changed: {n_changes}{extra}")
        else:
            hint = str(dest or path)
            extra = f" | names: {', '.join(changed_names)}" if n_changes else ""
            print(f"[DRY-RUN] would write {hint} | entries to change: {n_changes}{extra}")

    mode = "APPLY" if apply else "DRY-RUN"
    print(
        f"\n{mode} SUMMARY: parquet scanned={total_parquet}, parquet affected={affected_parquet}, "
        f"entries changed={changed_entries}, other files copied={copied_other}"
    )


if __name__ == "__main__":
    main()
    
# python scripts/fix_parquet_hf_metadata.py --root /data/robotwin/pi0_data/beat_block_hammer-50ep-agilex-demo_clean --out dataset --apply
