#!/usr/bin/env python3
"""Fix parquet metadata to replace 'List' type with 'Sequence' type for datasets compatibility."""

import json
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


def fix_parquet_metadata(file_path: Path):
    """Fix the metadata in a single parquet file."""
    table = pq.read_table(file_path)
    
    # Read existing metadata
    metadata = table.schema.metadata
    if metadata is None or b'huggingface' not in metadata:
        print(f"No huggingface metadata found in {file_path}")
        return
    
    # Parse the huggingface metadata
    hf_metadata_str = metadata[b'huggingface'].decode('utf-8')
    hf_metadata = json.loads(hf_metadata_str)
    
    # Fix the feature types: Replace 'List' with 'Sequence'
    features = hf_metadata.get('info', {}).get('features', {})
    modified = False
    
    for feature_name, feature_config in features.items():
        if feature_config.get('_type') == 'List':
            feature_config['_type'] = 'Sequence'
            modified = True
            print(f"  Fixed {feature_name}: List -> Sequence")
    
    if not modified:
        print(f"  No modifications needed for {file_path}")
        return
    
    # Update the metadata
    new_hf_metadata_str = json.dumps(hf_metadata)
    new_metadata = {b'huggingface': new_hf_metadata_str.encode('utf-8')}
    
    # Create new schema with updated metadata
    new_schema = table.schema.with_metadata(new_metadata)
    
    # Create new table with updated schema
    new_table = table.replace_schema_metadata(new_metadata)
    
    # Write back to the same file
    pq.write_table(new_table, file_path, compression='snappy')
    print(f"  Updated metadata for {file_path}")


def fix_directory(data_dir: Path):
    """Fix all parquet files in a directory."""
    parquet_files = list(data_dir.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to fix")
    
    for file_path in parquet_files:
        print(f"Processing {file_path}")
        try:
            fix_parquet_metadata(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_parquet_metadata.py <data_directory>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"Directory does not exist: {data_dir}")
        sys.exit(1)
    
    print(f"Fixing parquet metadata in: {data_dir}")
    fix_directory(data_dir)
    print("Done!")


if __name__ == "__main__":
    main()