import pandas as pd
import numpy as np
from pathlib import Path

# Base paths for the dataset structure[cite: 4]
DATA_PATH = Path(__file__).parent.parent / "datasets" / "mhealth"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"
OUTPUT_FILE = OUTPUT_PATH / "mhealth.csv"

# Dataset-specific constants[cite: 1]
SAMPLING_RATE_HZ = 50
TIME_INTERVAL_MS = int((1.0 / SAMPLING_RATE_HZ) * 1000)

# Activity mapping to align with README conventions[cite: 1, 5]
ACTIVITY_MAP = {
    0: 'non-study activity',     # Null class[cite: 1] mapped to Karas convention[cite: 5]
    1: 'standing',               # Standing still[cite: 1] mapped to HuGaDB convention[cite: 5]
    2: 'sitting',                # Sitting and relaxing[cite: 1] mapped to HuGaDB convention[cite: 5]
    3: 'lying_down',             # Lying down[cite: 1]
    4: 'flat',                   # Walking[cite: 1] mapped to flat (like Bruno dataset)[cite: 5]
    5: 'stairs_up',              # Climbing stairs[cite: 1] mapped to stairs_up[cite: 5]
    6: 'waist_bends_forward',    # Waist bends forward[cite: 1]
    7: 'frontal_elevation_arms', # Frontal elevation of arms[cite: 1]
    8: 'knees_bending',          # Knees bending (crouching)[cite: 1]
    9: 'bicycling',              # Cycling[cite: 1] mapped to bicycling[cite: 5]
    10: 'jogging',               # Jogging[cite: 1]
    11: 'running',               # Running[cite: 1] mapped to HuGaDB convention[cite: 5]
    12: 'jump_front_back'        # Jump front & back[cite: 1]
}

# Column names applying the naming convention from the README[cite: 5]
COLUMN_NAMES = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',               
    'chest_ecg_1', 'chest_ecg_2',                              
    'left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z',
    'left_ankle_gyr_x', 'left_ankle_gyr_y', 'left_ankle_gyr_z',
    'left_ankle_mag_x', 'left_ankle_mag_y', 'left_ankle_mag_z',
    'right_lower_acc_x', 'right_lower_acc_y', 'right_lower_acc_z',
    'right_lower_gyr_x', 'right_lower_gyr_y', 'right_lower_gyr_z',
    'right_lower_mag_x', 'right_lower_mag_y', 'right_lower_mag_z',
    'surface' 
]

def load_and_format_trial(file_path: Path, sid: int) -> pd.DataFrame:
    """
    Loads a single subject's log file, converts units, enforces a linear time sequence,
    maps integer labels to string labels, and formats columns.
    
    Args:
        file_path (Path): Path to the subject's .log file.
        sid (int): Subject ID.
        
    Returns:
        pd.DataFrame: Formatted dataframe structured with `time_ms`, `sid`, `surface` first.
    """
    # Load the raw tab-separated values[cite: 3]
    df = pd.read_csv(file_path, sep='\t', header=None, names=COLUMN_NAMES)
    
    # Map the integer labels to the string conventions[cite: 1, 5]
    df['surface'] = df['surface'].map(ACTIVITY_MAP)
    
    # 1. Enforce Linear Time Increase & Fill Missing Packets
    # The dataset is 50Hz, so we generate a strict 20ms timedelta index[cite: 1]
    df['time'] = pd.to_timedelta(df.index * TIME_INTERVAL_MS, unit='ms')
    df.set_index('time', inplace=True)
    
    # Resampling strictly enforces the linear time increase.
    expected_freq = f"{TIME_INTERVAL_MS}ms"
    df = df.resample(expected_freq).asfreq()
    
    # Reset the index to generate the required `time_ms` column[cite: 5]
    df.reset_index(inplace=True)
    df['time_ms'] = (df['time'].dt.total_seconds() * 1000).astype(int)
    
    # 2. Re-assign Subject ID (sid) and Forward-fill metadata 
    # (Since resampling introduces NaNs, we ensure sid and surface metadata persists)[cite: 5]
    df['sid'] = sid
    df['surface'] = df['surface'].ffill()
    
    # 3. Convert Units
    # Acceleration is natively in m/s^2[cite: 1]
    # Gyroscope data is in deg/s and must be converted to rad/s[cite: 1]
    gyro_cols = [col for col in df.columns if 'gyr' in col]
    df[gyro_cols] = df[gyro_cols] * (np.pi / 180.0)
    
    # 4. Reorder Columns
    # First columns must strictly be time_ms, sid, surface followed by the data[cite: 5]
    data_cols = [c for c in COLUMN_NAMES if c != 'surface']
    final_cols = ['time_ms', 'sid', 'surface'] + data_cols
    
    return df[final_cols]

def build_dataset_merger(data_dir: Path, output_filepath: Path):
    """
    Iterates through the mHealth dataset directory, processes all subjects, 
    and merges them into a single comprehensive CSV dataset.
    
    Args:
        data_dir (Path): The root directory containing the mHealth .log files.
        output_filepath (Path): The file path where the merged CSV will be saved.
    """
    all_trials = []
    
    # Ensure the output directory exists
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Iterate through subject files 1 to 10 as structured in the dataset tree[cite: 2]
    for sid in range(1, 11):
        file_name = f"mHealth_subject{sid}.log"
        file_path = data_dir / file_name
        
        if file_path.exists():
            print(f"Processing {file_name}...")
            subject_df = load_and_format_trial(file_path, sid)
            all_trials.append(subject_df)
        else:
            print(f"Warning: {file_name} not found in directory.")
            
    # Concatenate all subjects and save
    if all_trials:
        merged_df = pd.concat(all_trials, ignore_index=True)
        merged_df.to_csv(output_filepath, index=False)
        print(f"Dataset successfully merged and saved to {output_filepath}")
    else:
        print("No data files were processed.")

if __name__ == "__main__":
    build_dataset_merger(DATA_PATH, OUTPUT_FILE)