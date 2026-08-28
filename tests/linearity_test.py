from pathlib import Path
import os
import re
import pandas as pd
import numpy as np

OUTPUT_PATH = Path(__file__).parent.parent / "outputs"
DATASETS_YAML_PATH = Path(__file__).parent.parent / "datasets.yaml"


def resolve_dataset_path(default_filename: str) -> Path:
    dataset_name = os.environ.get("DATASET_NAME")
    if dataset_name:
        return OUTPUT_PATH / f"{dataset_name}.csv"
    return OUTPUT_PATH / default_filename


def resolve_dataset_name(default_filename: str) -> str:
    dataset_name = os.environ.get("DATASET_NAME")
    if dataset_name:
        return dataset_name
    return Path(default_filename).stem


def load_dataset_fs(dataset_name: str) -> float:
    current_dataset = None
    fs_value = None

    for raw_line in DATASETS_YAML_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        dataset_match = re.match(r"^\s{2}([A-Za-z0-9_\-]+):\s*$", line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            continue

        if current_dataset != dataset_name:
            continue

        fs_match = re.match(r"^\s{4}fs:\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if fs_match:
            fs_value = float(fs_match.group(1))
            break

    if fs_value is None:
        raise ValueError(f"Could not find fs for dataset '{dataset_name}' in {DATASETS_YAML_PATH.name}")

    return fs_value

def check_time_linearity(time_ms, sid, fs, trials=None, sort_first=False, tolerance=1):
    """
    Checks if time sequences for each subject ID (and optionally trial) are strictly linear.
    
    Parameters:
    - time_ms: Array of timestamps
    - sid: Array of subject/session IDs
    - trials: Optional Array of trial numbers
    - sort_first: Boolean, whether to sort the timestamps before checking diffs
    - tolerance: Float, allowable variance in step size (useful for float timestamps)
    
    Returns:
    - dict: A dictionary mapping SID (or a tuple of (SID, Trial)) to a boolean 
            (True if linear, False if not)
    """
    results = {}
    
    # Drop NaNs if they exist to prevent diff calculation errors
    valid_mask = ~np.isnan(time_ms)
    time_ms = time_ms[valid_mask]
    sid = sid[valid_mask]
    
    # Apply the same mask to trials to ensure array lengths match
    if trials is not None:
        trials = trials[valid_mask]

    for s in np.unique(sid):
        # Create a list of groups to check. 
        # If trials exist, we group by (SID, Trial). Otherwise, just SID.
        sub_groups = []
        if trials is not None:
            trials_for_sid = np.unique(trials[sid == s])
            for t in trials_for_sid:
                mask = (sid == s) & (trials == t)
                sub_groups.append(((s, t), time_ms[mask]))
        else:
            sub_groups.append((s, time_ms[sid == s]))
            
        for key, t_ms in sub_groups:
            # Edge case 1: Insufficient data points
            if len(t_ms) < 2:
                print(f"Key: {key} has fewer than 2 points. (Considered Linear by default)")
                results[key] = True
                continue
            
            # Edge case 2: Unsorted data
            if sort_first:
                t_ms = np.sort(t_ms)
            
            diff = np.diff(t_ms)
            
            # Edge case 3: Floating point inconsistencies
            expected_diff = ( 1 / fs ) * 1000

            # indices = np.where(diff < 0)[0]
            # print(len(diff), len(t_ms))
            # print(t_ms[:10])
            # print(indices[:10])
            # print(t_ms[1341], t_ms[1342], t_ms[1343])
            # print(np.array(diff - expected_diff)[:10])
            # print(t_ms[198:205])
            # for idx in indices:
            #     print(idx)

            # print(np.min(diff), np.max(diff), expected_diff, abs(np.min(diff) - expected_diff), abs(np.max(diff) - expected_diff))
            
            if np.min(diff) < 0:
                print(f"Non-linear time detected for Key: {key} (Negative diff found)")
                results[key] = False
            if np.any(np.abs(diff - expected_diff) > tolerance):
                print(f"Non-linear time detected for Key: {key} (Diff outside tolerance)")
                results[key] = False
            else:
                print(f"Time is linear for Key: {key}")
                results[key] = True
                # for idx in indices:
                #     print(f'idx:{idx}\n\t t_ms[{idx-1}]:{t_ms[idx-1]}\n\t t_ms[{idx}]:{t_ms[idx]}\n\t t_ms[{idx+1}]:{t_ms[idx+1]}')
            
    return results


def load_data(file_path):
    '''
    return (tuple): time_ms, sid, trials (or None if no trial column exists)
    '''
    df = pd.read_csv(file_path)
    time_ms = df['time_ms'].to_numpy()
    sid = df['sid'].to_numpy()
    
    # Safely extract trial_no if it exists in the dataset
    trials = df['trial_no'].to_numpy() if 'trial_no' in df.columns else None
    
    return time_ms, sid, trials

beach_path = resolve_dataset_path("beach.csv")
bruno_path = resolve_dataset_path("bruno.csv")
hugadb_path = resolve_dataset_path("hugadb.csv")
luo_path = resolve_dataset_path("luo.csv")
karas_path = resolve_dataset_path("karas.csv")

dataset_name = resolve_dataset_name("bruno.csv")
fs = load_dataset_fs(dataset_name)
dataset_path = resolve_dataset_path(f"{dataset_name}.csv")

# Load the data and unpack the tuple
time_data, sid_data, trial_data = load_data(dataset_path)

# Run the updated check
results_dict = check_time_linearity(time_data, sid_data, fs, trials=trial_data)