import pandas as pd
from pathlib import Path
from typing import List

# Standard columns updated to the {sensor_loc}_{Metric}_{x/y/z} convention[cite: 4].
# Sensor locations borrow established names from the Luo and Beach datasets[cite: 4].
HUGADB_COLUMNS = [
    'right_foot_Acc_x', 'right_foot_Acc_y', 'right_foot_Acc_z', 'right_foot_Gyr_x', 'right_foot_Gyr_y', 'right_foot_Gyr_z',
    'right_shank_Acc_x', 'right_shank_Acc_y', 'right_shank_Acc_z', 'right_shank_Gyr_x', 'right_shank_Gyr_y', 'right_shank_Gyr_z',
    'right_thigh_Acc_x', 'right_thigh_Acc_y', 'right_thigh_Acc_z', 'right_thigh_Gyr_x', 'right_thigh_Gyr_y', 'right_thigh_Gyr_z',
    'left_foot_Acc_x', 'left_foot_Acc_y', 'left_foot_Acc_z', 'left_foot_Gyr_x', 'left_foot_Gyr_y', 'left_foot_Gyr_z',
    'left_shank_Acc_x', 'left_shank_Acc_y', 'left_shank_Acc_z', 'left_shank_Gyr_x', 'left_shank_Gyr_y', 'left_shank_Gyr_z',
    'left_thigh_Acc_x', 'left_thigh_Acc_y', 'left_thigh_Acc_z', 'left_thigh_Gyr_x', 'left_thigh_Gyr_y', 'left_thigh_Gyr_z',
    'right_thigh_EMG', 'left_thigh_EMG', 'ActivityID'
]

# Mapping of HuGaDB Activity IDs to the standardized 'surface' categories used across your sets (e.g., 'flat', 'stairs_up', 'driving')[cite: 4].
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


def parse_hugadb_file(filepath: Path) -> pd.DataFrame:
    """
    Reads a single HuGaDB text file, skipping metadata headers, and appends 
    standardized base columns.

    Parameters:
        filepath (Path): The Path object pointing to the specific .txt file.

    Returns:
        pd.DataFrame: A formatted dataframe with assigned columns and standardized surface names.
    """
    filename_no_ext = filepath.stem
    parts = filename_no_ext.split('_')
    
    # Extract participant ID and trial number based on the trailing components.
    trial_no = int(parts[-1])
    sid = int(parts[-2])
    
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        comment='#', 
        header=None, 
        names=HUGADB_COLUMNS
    )
    
    # Append the required base project columns[cite: 4].
    # time_ms is initialized as a generic sequence since raw timestamp formats vary; update calculation if a specific frequency is needed.
    df['time_ms'] = range(len(df))
    df['sid'] = sid
    df['surface'] = df['ActivityID'].map(ACTIVITY_TO_SURFACE).fillna('other')
    df['trial_no'] = trial_no
    
    # Drop the ActivityID column as it is now redundant with the standard 'surface' column.
    df = df.drop(columns=['ActivityID'])
    
    return df


def merge_directory(dir_path: Path, output_filename: Path) -> None:
    """
    Iterates through all text files in a specified directory, parses them, 
    and concatenates them into a single CSV.

    Parameters:
        dir_path (Path): Directory containing the HuGaDB text files.
        output_filename (Path): Path to the generated CSV file.
    """
    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return

    print(f"Processing files in {dir_path}...")
    
    txt_files = list(dir_path.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name.lower() != 'readme.txt']

    dataframes: List[pd.DataFrame] = []
    
    for file in txt_files:
        df = parse_hugadb_file(file)
        dataframes.append(df)
        
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.to_csv(output_filename, index=False)
        print(f"Successfully saved merged data to {output_filename} ({len(merged_df)} rows).")
    else:
        print(f"No valid data files found in {dir_path}.")


def main():
    """
    Main orchestrator to merge v1 and v2 datasets into two independent CSV files.
    """
    # Define the root of the dataset structure.
    base_dataset_path = Path(__file__).parent.parent / 'datasets' / 'hugadb'
    output_dir = Path(__file__).parent.parent / 'outputs'
    
    # Map the precise directory paths shown in the tree structure[cite: 2].
    v1_data_dir = base_dataset_path / "v1" / "Data"
    v2_data_dir = base_dataset_path / "v2"
    
    # Merge v1 and v2 files separately
    merge_directory(v1_data_dir, output_dir / "hugadbv1.csv")
    merge_directory(v2_data_dir, output_dir / "hugadbv2.csv")


if __name__ == "__main__":
    main()