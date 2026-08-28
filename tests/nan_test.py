from pathlib import Path
import os
import pandas as pd
import numpy as np

OUTPUT_PATH = Path(__file__).parent.parent / "outputs"


def resolve_dataset_path(default_filename: str) -> Path:
    dataset_name = os.environ.get("DATASET_NAME")
    if dataset_name:
        return OUTPUT_PATH / f"{dataset_name}.csv"
    return OUTPUT_PATH / default_filename

def check_nans_in_chunks(file_path: Path, chunksize: int = 500000, context_window: int = 2, max_examples: int = 3):
    """
    Reads a large CSV file in chunks, checks for NaN values across all columns,
    and prints small contextual sections around rows containing NaNs.

    Parameters:
    - file_path: Path object pointing to the CSV file.
    - chunksize: Integer, number of rows to load into memory per chunk. 
                 Keeps RAM usage low for multi-gigabyte files.
    - context_window: Integer, number of rows to display before and after the NaN row.
    - max_examples: Integer, maximum number of NaN occurrences to print per chunk 
                    to avoid flooding the console output.

    Returns:
    - None
    """
    print(f"--- Checking for NaNs in {file_path.name} ---")
    
    total_nans_per_col = pd.Series(dtype=int)
    total_rows = 0
    has_nans = False
    
    try:
        # Load the data per chunks to handle large files
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
            total_rows += len(chunk)
            
            # Update total NaN counts per column for the final summary
            chunk_nans = chunk.isna().sum()
            total_nans_per_col = total_nans_per_col.add(chunk_nans, fill_value=0)
            
            # Find rows that contain at least one NaN
            row_has_nan = chunk.isna().any(axis=1)
            
            if row_has_nan.any():
                has_nans = True
                print(f"\n[Chunk {chunk_idx + 1}] Found NaNs in rows {chunk.index[0]} to {chunk.index[-1]}")
                
                # Get the exact integer positions (iloc) of rows with NaNs in this chunk
                nan_positions = np.where(row_has_nan)[0]
                
                # Limit the number of examples printed per chunk
                for count, pos in enumerate(nan_positions):
                    if count >= max_examples:
                        print(f"  ... and {len(nan_positions) - max_examples} more NaN rows in this chunk.")
                        break
                        
                    # Calculate context window bounds (ensuring we don't index out of bounds)
                    start_pos = max(0, pos - context_window)
                    end_pos = min(len(chunk), pos + context_window + 1)
                    
                    # Extract the small section of data
                    context_df = chunk.iloc[start_pos:end_pos]
                    
                    # Identify exactly which columns have NaNs in the target row
                    target_row = chunk.iloc[pos]
                    nan_cols = target_row.index[target_row.isna()].tolist()
                    
                    print(f"\n  -> Example at Global Row Index {chunk.index[pos]}:")
                    print(f"  -> NaN found in column(s): {nan_cols}")
                    print(context_df.to_string())
                    print("-" * 60)
                    
        if not has_nans:
            print("No NaNs found in the entire dataset. It is perfectly clean!")
        else:
            print("\n=== Final NaN Summary ===")
            final_nans = total_nans_per_col[total_nans_per_col > 0].astype(int)
            print(final_nans.to_string())
            print(f"\nTotal Rows Processed: {total_rows:,}")
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Define dataset paths
    beach_path = resolve_dataset_path("beach.csv")
    bruno_path = resolve_dataset_path("bruno.csv")
    hugadb_path = resolve_dataset_path("hugadb.csv")
    luo_path = resolve_dataset_path("luo.csv")
    karas_path = resolve_dataset_path("karas.csv")

    # Run the chunked NaN checker
    # (Update the variable to check different datasets)
    check_nans_in_chunks(luo_path)