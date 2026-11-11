import argparse
import ast
import os
import pandas as pd
from pathlib import Path

def convert_csv_to_parquet(input_csv, output_parquet):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Parse the urls column
    print("Parsing URLs...")
    df['urls'] = df['urls'].apply(ast.literal_eval)

    # Take the first URL
    print("Selecting first URL per prompt...")
    df['url'] = df['urls'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    
    # Filter out rows with no URL if any
    initial_len = len(df)
    df = df.dropna(subset=['url'])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with no valid URLs.")

    # Rename prompt to caption
    print("Renaming columns...")
    df = df.rename(columns={'prompt': 'caption'})

    # Create new index
    print("Creating new index...")
    df = df.reset_index(drop=True)
    df['index'] = df.index

    # Add overfit_type
    print("Adding overfit_type column...")
    df['overfit_type'] = "MV"

    # Select and reorder columns
    cols = ['index', 'url', 'caption', 'overfit_type']
    final_df = df[cols]

    print(f"Saving to {output_parquet}...")
    final_df.to_parquet(output_parquet)
    print("Done!")
    
    # Verification
    print("\nVerifying output:")
    loaded_df = pd.read_parquet(output_parquet)
    print(f"Loaded columns: {loaded_df.columns.tolist()}")
    print(f"Row count: {len(loaded_df)}")
    print(f"Sample row:\n{loaded_df.iloc[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-parquet", type=str, required=True)
    args = parser.parse_args()

    base_dir = Path(args.output_parquet).parent
    os.makedirs(base_dir, exist_ok=True)
    
    convert_csv_to_parquet(args.input_csv, args.output_parquet)
