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

def print_metrics(file_path: Path):
    """
    Prints the unique metrics present in a CSV file.

    Parameters:
    - file_path: Path object pointing to the CSV file.
    """
    try:
        df = pd.read_csv(file_path, nrows=0)  # Read only the header
        # possible: "Acc","Gyr","EMG" in column name. eg column name: "left_thigh_EMG" or "right_wrist_Acc_x"
        unique_metrics = set()
        for col in df.columns:
            if col in ["time_ms", "sid", "surface", "trial_no", "packetcounter"]:
                continue  # Skip metadata columns
            split1 = col.split('_')[-1]
            split2 = col.split('_')[-2]
            unique_metrics.add(split1 if split1 not in ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'] else split2)  # Extract the metric part

        print(f"Unique metrics in {file_path.name}:")
        for metric in unique_metrics:
            print(f" - {metric}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# --- Execution ---
if __name__ == "__main__":
    paths = list(resolve_dataset_path(name) for name in ["beach.csv", "bruno.csv", "hugadb.csv", "luo.csv", "karas.csv"])

    # Run the chunked NaN checker
    # (Update the variable to check different datasets)
    for path in paths:
        print_metrics(path)