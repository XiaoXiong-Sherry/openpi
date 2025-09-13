#!/usr/bin/env python3
"""
Convert Hugging Face parquet metadata entries with `_type: "List"` to `_type: "Sequence"`.

This updates only the JSON stored in the `huggingface` key of Parquet file metadata.
It preserves all other metadata and data. Writes atomically via a temp file + rename.

Usage:
  python scripts/convert_parquet_hf_list_to_sequence.py \
      --root /path/to/dataset --dry-run

  python scripts/convert_parquet_hf_list_to_sequence.py \
      --root /path/to/dataset --apply
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def iter_all_paths(root: Path):
    """Yield all files and directories under root (depth-first)."""
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Yield directory first so caller can create it
        yield d, True
        for fname in filenames:
            yield d / fname, False


def load_hf_metadata(pq_path: Path):
    md = pq.read_metadata(str(pq_path))
    meta = md.metadata or {}
    hf_raw = meta.get(b"huggingface")
    if not hf_raw:
        return None, meta
    try:
        hf = json.loads(hf_raw.decode("utf-8"))
    except Exception:
        return None, meta
    return hf, meta


def convert_list_to_sequence(hf: dict) -> tuple[dict, int]:
    """Return updated hf dict and number of changes made."""
    if not isinstance(hf, dict):
        return hf, 0
    info = hf.get("info") or {}
    feats = info.get("features")
    if not isinstance(feats, dict):
        return hf, 0
    changes = 0
    for name, spec in feats.items():
        if isinstance(spec, dict) and spec.get("_type") == "List":
            # Change only the type; keep feature/value and length as-is
            spec["_type"] = "Sequence"
            changes += 1
    if changes:
        # Ensure the structure is preserved
        hf.setdefault("info", {}).setdefault("features", feats)
    return hf, changes


def rewrite_parquet_with_metadata(src: Path, new_metadata: dict):
    # Read table
    table = pq.read_table(str(src))
    # Merge new metadata with existing schema metadata
    schema = table.schema
    existing = dict(schema.metadata or {})
    existing.update(new_metadata)
    new_table = table.replace_schema_metadata(existing)
    # Write to temp and atomically replace
    with tempfile.NamedTemporaryFile(dir=str(src.parent), suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(new_table, str(tmp_path))
        os.replace(str(tmp_path), str(src))
    except Exception:
        # Cleanup temp on failure
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="Source dataset root directory")
    ap.add_argument("--out", type=Path, default=None, help="Output dataset root (mirror). If omitted and --apply is set, modifies in-place.")
    ap.add_argument("--apply", action="store_true", help="Apply changes; otherwise dry-run")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of parquet files processed (debug)")
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    src_root = root.resolve()
    out_root = args.out.resolve() if args.out else None

    total_parquet = 0
    affected_parquet = 0
    changed_entries = 0
    copied_other = 0

    import shutil

    # Walk and mirror
    parquet_seen = 0
    for path, is_dir in iter_all_paths(src_root):
        rel = path.relative_to(src_root)
        if is_dir:
            if out_root:
                dest_dir = out_root / rel
                if args.apply:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"[DRY-RUN] mkdir {dest_dir}")
            continue

        # file
        if path.suffix == ".parquet":
            if args.limit is not None and parquet_seen >= args.limit:
                continue
            parquet_seen += 1
            total_parquet += 1
            hf, meta = load_hf_metadata(path)
            if hf is None:
                # No HF metadata; copy-through
                if out_root:
                    dest = out_root / rel
                    if args.apply:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                        print(f"[APPLIED] copy {path} -> {dest} (no hf meta)")
                    else:
                        print(f"[DRY-RUN] copy {path} -> {dest} (no hf meta)")
                else:
                    # In-place, nothing to change
                    pass
                continue

            updated_hf, n_changes = convert_list_to_sequence(hf)
            if n_changes == 0:
                # No change; copy-through if out_root is set
                if out_root:
                    dest = out_root / rel
                    if args.apply:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                        print(f"[APPLIED] copy {path} -> {dest} (no change)")
                    else:
                        print(f"[DRY-RUN] copy {path} -> {dest} (no change)")
                continue

            affected_parquet += 1
            changed_entries += n_changes
            if args.apply:
                new_meta = dict(meta)
                new_meta[b"huggingface"] = json.dumps(updated_hf, ensure_ascii=False).encode("utf-8")
                # Determine destination path
                if out_root:
                    dest = out_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # Write directly to destination
                    table = pq.read_table(str(path))
                    schema = table.schema
                    existing = dict(schema.metadata or {})
                    existing.update(new_meta)
                    new_table = table.replace_schema_metadata(existing)
                    pq.write_table(new_table, str(dest))
                    print(f"[APPLIED] write {dest} | entries changed: {n_changes}")
                else:
                    rewrite_parquet_with_metadata(path, new_meta)
                    print(f"[APPLIED] {path} | entries changed: {n_changes}")
            else:
                dest_hint = str((out_root / rel) if out_root else path)
                print(f"[DRY-RUN] would write {dest_hint} | entries to change: {n_changes}")
        else:
            # Non-parquet file: copy-through if out_root
            if out_root:
                dest = out_root / rel
                if args.apply:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                copied_other += 1
                if args.apply:
                    print(f"[APPLIED] copy {path} -> {dest}")
                else:
                    print(f"[DRY-RUN] copy {path} -> {dest}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"\n{mode} SUMMARY: parquet scanned={total_parquet}, parquet affected={affected_parquet}, "
        f"entries changed={changed_entries}, other files copied={copied_other}"
    )


if __name__ == "__main__":
    main()
