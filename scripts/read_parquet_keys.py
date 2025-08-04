
import polars as pl
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the Parquet file.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to print.")
    args = parser.parse_args()

    try:
        df = pl.read_parquet(args.file_path)
        if "observation.state" in df.columns:
            print(f"First {args.rows} values of 'observation.state':")
            print(df.select("observation.state").head(args.rows))
        else:
            print("Column 'observation.state' not found.")
            print(f"Available columns are: {df.columns}")

    except Exception as e:
        print(f"Error reading Parquet file: {e}")

if __name__ == "__main__":
    main()
