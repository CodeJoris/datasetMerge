"""
HuGaDB Dataset Merge Script
    - locations: ['right_foot', 'right_thigh', 'left_foot', 'right_shank', 'left_shank', 'left_thigh']
    - metrics: ['acc', 'gyr', 'emg']
    - units: ['int_16', 'int_16', 'uint_8']
    - Conversion line: line 156 
    - Output units: ['m/s^2', 'rad/s', 'uint_8']    
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

# Standard columns updated to the {sensor_loc}_{Metric}_{x/y/z} convention.
# The main data body of every file has 39 columns in a fixed order.
HUGADB_COLUMNS = [
    'right_foot_acc_x', 'right_foot_acc_y', 'right_foot_acc_z', 'right_foot_gyr_x', 'right_foot_gyr_y', 'right_foot_gyr_z',
    'right_shank_acc_x', 'right_shank_acc_y', 'right_shank_acc_z', 'right_shank_gyr_x', 'right_shank_gyr_y', 'right_shank_gyr_z',
    'right_thigh_acc_x', 'right_thigh_acc_y', 'right_thigh_acc_z', 'right_thigh_gyr_x', 'right_thigh_gyr_y', 'right_thigh_gyr_z',
    'left_foot_acc_x', 'left_foot_acc_y', 'left_foot_acc_z', 'left_foot_gyr_x', 'left_foot_gyr_y', 'left_foot_gyr_z',
    'left_shank_acc_x', 'left_shank_acc_y', 'left_shank_acc_z', 'left_shank_gyr_x', 'left_shank_gyr_y', 'left_shank_gyr_z',
    'left_thigh_acc_x', 'left_thigh_acc_y', 'left_thigh_acc_z', 'left_thigh_gyr_x', 'left_thigh_gyr_y', 'left_thigh_gyr_z',
    'right_thigh_emg', 'left_thigh_emg', 'activityid'
]

# Mapping of HuGaDB Activity IDs to the standardized 'surface' categories.
ACTIVITY_TO_SURFACE = {
    1: 'flat',          
    2: 'flat',          
    3: 'stairs_up',     
    4: 'stairs_down',   
    5: 'sitting',       
    6: 'sitting_down',  
    7: 'standing_up',   
    8: 'standing',      
    9: 'bicycling',     
    10: 'elevator_up',  
    11: 'elevator_down',
    12: 'driving'       
}

def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw 16-bit integer sensor data to physical units.
    
    Accelerometers are scaled from -2g to 2g, and gyroscopes from -2000 to 2000 deg/sec.
    Values are encoded as int16 datatypes (-32768 to 32767).
    This function converts accelerometer values to m/s^2 and gyroscope values to rad/s.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the raw accelerometer and gyroscope columns.
        
    Returns:
        pd.DataFrame: The dataframe updated with standardized physical unit values.
    """
    g_to_ms2 = 9.80665
    # Calculate multipliers to shift from 16-bit int bounds to respective unit bounds
    acc_scale = (2.0 * g_to_ms2) / 32768.0
    gyr_scale = (2000.0 * np.pi / 180.0) / 32768.0
    
    # Identify which columns belong to which sensor
    acc_cols = [col for col in df.columns if 'acc' in col]
    gyr_cols = [col for col in df.columns if 'gyr' in col]
    
    # Apply conversions
    df[acc_cols] = df[acc_cols] * acc_scale
    df[gyr_cols] = df[gyr_cols] * gyr_scale
    
    return df

def enforce_linear_time(df: pd.DataFrame, time_col: str, step_ms: int = 17) -> pd.DataFrame:
    """
    Enforces a strict linear time increase for a given time column within a trial.
    
    This function finds the minimum and maximum values of the specified index column 
    and reindexes the DataFrame to include every step interval in that range. Any 
    missing packets in the data will be filled with NaNs to maintain temporal alignment.

    Parameters:
        df (pd.DataFrame): The single-trial dataset.
        time_col (str): The column name representing the time index.
        step_ms (int): The exact time increment between packets in milliseconds.

    Returns:
        pd.DataFrame: A linearly continuous DataFrame with NaNs filling any missing intervals.
    """
    if df.empty or time_col not in df.columns:
        return df

    # Ensure the tracking column is an integer
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df[time_col] = df[time_col].astype(int)

    # Determine global start and end points for this trial
    min_val = df[time_col].min()
    max_val = df[time_col].max()

    # Create an unbroken linear sequence utilizing the hardware's 17 ms step
    complete_index = range(min_val, max_val + 1, step_ms)

    # Reindex to enforce linear time, padding missing rows with NaNs
    df = df.set_index(time_col)
    df_reindexed = df.reindex(complete_index)

    # Restore the tracking column from the index
    df_reindexed = df_reindexed.reset_index()
    df_reindexed = df_reindexed.rename(columns={'index': time_col})

    return df_reindexed

def parse_hugadb_file(filepath: Path) -> pd.DataFrame:
    """
    Reads a single HuGaDB text file, appends standardized base columns, extracts 
    metadata from the filename, scales unit types, and enforces a linear packet sequence.

    The file naming convention follows HGD_vX_ACT_PR_CNT.txt. Lines beginning 
    with '#' are treated as headers and skipped.

    Parameters:
        filepath (Path): The Path object pointing to the specific .txt file.

    Returns:
        pd.DataFrame: A formatted DataFrame with assigned columns, standardized 
                      surface names, normalized physical units, and padded missing packets (NaNs).
    """
    filename_no_ext = filepath.stem
    parts = filename_no_ext.split('_')
    
    # Extract participant ID (PR) and trial number (CNT) based on the file template
    trial_no = int(parts[-1])
    sid = int(parts[-2])
    
    # Use header=0 to properly assign the column names and prevent type errors
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        comment='#', 
        header=0, 
        names=HUGADB_COLUMNS
    )
    
    # Base hardware rate dictates exactly 17 ms increments between logged rows
    df['time_ms'] = df.index * 17
    
    # Append the required base project metadata columns
    df['sid'] = sid
    df['surface'] = df['activityid'].map(ACTIVITY_TO_SURFACE).fillna('other')
    df['trial_no'] = trial_no
    
    # Drop the ActivityID column as it is now redundant with 'surface'
    df = df.drop(columns=['activityid'])
    
    # Convert hardware integer counts to standard physical metrics (m/s^2 & rad/s)
    df = convert_units(df)
    
    # Enforce linear time increase and pad missing packets with NaNs
    df = enforce_linear_time(df, time_col='time_ms', step_ms=17)
    
    return df

def merge_directory(dir_path: Path, output_filename: Path) -> None:
    """
    Iterates through all text files in a specified directory structure, parses them, 
    enforces continuity, and concatenates them into a single CSV.

    Files named 'readme.txt' or 'README.md' are intentionally skipped.

    Parameters:
        dir_path (Path): Directory containing the HuGaDB text files.
        output_filename (Path): Path to the generated output CSV file.
    """
    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return

    print(f"Processing files in {dir_path}...")
    
    # Traverse directory and filter out README files
    txt_files = list(dir_path.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name.lower() not in ['readme.txt', 'readme.md']]

    dataframes: List[pd.DataFrame] = []
    
    for file in txt_files:
        df = parse_hugadb_file(file)
        dataframes.append(df)
        
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.to_csv(output_filename, index=False, na_rep='')
        print(f"Successfully saved merged data to {output_filename} ({len(merged_df)} rows).")
    else:
        print(f"No valid data files found in {dir_path}.")

def main():
    """
    Main orchestrator to traverse the directory tree and merge the datasets.
    
    Targets the v1 and v2 directories explicitly, matching the nested Data folder
    for v1, and generates independent CSV files without merging the separate datasets.
    """
    # Define the root of the dataset structure
    base_dataset_path = Path(__file__).parent.parent / 'datasets' / 'hugadb'
    output_dir = Path(__file__).parent.parent / 'outputs'
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map the precise directory paths shown in the tree structure
    v2_data_dir = base_dataset_path / "v2"
    
    # Process v1 and v2 files separately, keeping experimental datasets completely independent
    merge_directory(v2_data_dir, output_dir / "hugadb.csv")

if __name__ == "__main__":
    main()