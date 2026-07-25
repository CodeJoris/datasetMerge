''''
Check the existing columns in a given CSV file
'''

from pathlib import Path
import os
import pandas as pd


OUTPUT_PATH = Path(__file__).parent.parent / "outputs"


def resolve_dataset_path(default_filename: str) -> Path:
    dataset_name = os.environ.get("DATASET_NAME")
    if dataset_name:
        return OUTPUT_PATH / f"{dataset_name}.csv"
    return OUTPUT_PATH / default_filename

def print_columns(file_path: Path):
    """
    Prints the column names of a CSV file.

    Parameters:
    - file_path: Path object pointing to the CSV file.
    """
    try:
        df = pd.read_csv(file_path, nrows=0)  # Read only the header
        print(f"Columns in {file_path.name}:")
        for col in df.columns:
            print(f" - {col}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# --- Execution ---
if __name__ == "__main__":
    # Define dataset paths[cite: 7]
    beach_path = resolve_dataset_path("beach.csv")
    bruno_path = resolve_dataset_path("bruno.csv")
    hugadb_path = resolve_dataset_path("hugadb.csv")
    luo_path = resolve_dataset_path("luo.csv")
    karas_path = resolve_dataset_path("karas.csv")

    # Run the chunked NaN checker
    # (Update the variable to check different datasets)
    print_columns(luo_path)
    # print_metrics(karas_path)
