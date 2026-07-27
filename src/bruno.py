from importlib.resources import path
import os
import pandas as pd
from pathlib import Path

# --- Global Constants ---
SAMPLING_RATE_HZ = 32
MS_PER_SAMPLE = 1000 / SAMPLING_RATE_HZ
G_TO_MS2 = 9.80665

SURFACE_MAPPING = {
    'climb_stairs': 'stairs_up',
    'descend_stairs': 'stairs_down',
    'walk': 'flat'
}

def parse_filename(filename, sid_map, next_sid):
    """
    Parses the dataset filename to extract the surface type and standardizes the subject ID.

    Parameters
    ----------
    filename : str
        The name of the file (e.g., 'Accelerometer-2011-03-24-10-24-39-climb_stairs-f1.txt').
    sid_map : dict
        A dictionary mapping original volunteer IDs (e.g., 'f1') to integer SIDs.
    next_sid : int
        The next available integer to assign for a new volunteer ID.

    Returns
    -------
    tuple
        (surface (str), sid (int), updated_next_sid (int)) or (None, None, next_sid) if parsing fails.
    """
    parts = filename.replace('.txt', '').split('-')
    
    # Ensure the filename conforms to the expected [START_TIME]-[HMP]-[VOLUNTEER] format
    if len(parts) < 9:
        return None, None, next_sid
        
    original_hmp = parts[7]
    volunteer_id = parts[8]
    
    # Map the activity to the unified surface naming convention
    surface = SURFACE_MAPPING.get(original_hmp, original_hmp)
    
    # Map the string volunteer ID to a unique integer
    if volunteer_id not in sid_map:
        sid_map[volunteer_id] = next_sid
        next_sid += 1
        
    sid = sid_map[volunteer_id]
    
    return surface, sid, next_sid

def convert_acceleration_to_ms2(df):
    """
    Converts 6-bit coded acceleration values to standard gravity (g), then to m/s^2.
    Formula applied: real_val = -1.5g + (coded_val/63) * 3g.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing raw coded 6-bit acceleration data in columns 0, 1, and 2.

    Returns
    -------
    pd.DataFrame
        The dataframe with converted acceleration values.
    """
    for col in range(3):
        df[col] = (-1.5 + (df[col] / 63.0) * 3.0) * G_TO_MS2
    return df

def format_dataframe(df, surface, sid, trial_no):
    """
    Adds required metadata columns, enforces a strictly linear time sequence, 
    and renames existing columns to match the unified standard.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the converted x, y, and z acceleration data.
    surface : str
        The standardized surface name.
    sid : int
        The standardized subject ID.
    trial_no : int
        The sequential ID to differentiate between distinct trials.

    Returns
    -------
    pd.DataFrame
        The fully formatted dataframe ready for concatenation, with missing packets filled with NaNs.
    """
    # Enforce strict linear time increase by reindexing from the min to max packet.
    # If the original index skipped numbers (missing packets), pd.reindex fills those gaps with NaNs.
    min_idx = df.index.min()
    max_idx = df.index.max()
    linear_index = pd.RangeIndex(start=min_idx, stop=max_idx + 1)
    df = df.reindex(linear_index)
    
    # Create the time_ms column based on the 32 Hz sampling rate
    df['time_ms'] = (df.index * MS_PER_SAMPLE).astype(int)
    
    # Assign metadata
    df['trial_no'] = trial_no
    df['sid'] = sid
    df['surface'] = surface
    
    # Reorder and rename to match the unified database specifications
    df = df[['time_ms', 'sid', 'surface', 'trial_no', 0, 1, 2]]
    df.columns = [
        'time_ms',
        'sid', 
        'surface', 
        'trial_no', 
        'right_wrist_acc_x', 
        'right_wrist_acc_y', 
        'right_wrist_acc_z'
    ]
    return df

def process_single_file(file_path, filename, sid_map, next_sid, trial_no):
    """
    Reads and processes a single accelerometer text file.

    Parameters
    ----------
    file_path : str
        The full path to the file.
    filename : str
        The name of the file.
    sid_map : dict
        The current mapping of volunteer IDs to integer SIDs.
    next_sid : int
        The next available integer SID.
    trial_no : int
        The unique identifier for the specific trial being parsed.

    Returns
    -------
    tuple
        (processed_dataframe or None, updated_next_sid)
    """
    surface, sid, next_sid = parse_filename(filename, sid_map, next_sid)
    
    if surface is None:
        return None, next_sid
        
    try:
        # Read data; assuming space or comma-separated values with no header
        df = pd.read_csv(file_path, sep=r'\s+|,', header=None, engine='python', on_bad_lines='skip', skipinitialspace=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, next_sid
        
    df = convert_acceleration_to_ms2(df)
    df = format_dataframe(df, surface, sid, trial_no)
    
    return df, next_sid

def process_bruno_dataset(base_dir, output_csv):
    """
    Traverses the Bruno dataset directory, processes all valid files, and exports a single combined CSV.

    Parameters
    ----------
    base_dir : str or Path
        The root directory of the dataset (e.g., './datasets/bruno1').
    output_csv : str or Path
        The desired filename/path for the exported CSV file.
    """
    data_frames = []
    sid_map = {}
    next_sid = 1
    trial_no = 1  # Initialize trial counter

    # Traverse the directory tree
    for root, _, files in os.walk(base_dir):
        # Skip the duplicate _MODEL directories
        if root.endswith('_MODEL'):
            continue
            
        for file in files:
            # Skip documentation files or non-txt files
            if file.lower() in ['manual.txt', 'readme.txt'] or not file.endswith('.txt'):
                continue
                
            file_path = os.path.join(root, file)
            
            # Process the file and update the tracking IDs
            df, next_sid = process_single_file(file_path, file, sid_map, next_sid, trial_no)
            
            if df is not None:
                data_frames.append(df)
                trial_no += 1  # Increment only when a trial is successfully ingested

    # Combine all individual dataframes
    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Successfully combined {len(data_frames)} files into '{output_csv}'.")
        print(f"Total trials recorded: {trial_no - 1}")
        print(f"Volunteer SID mapping used: {sid_map}")
    else:
        print("No valid data files were found.")

if __name__ == "__main__":
    # Target directories
    dataset_dir = Path(__file__).parent.parent / 'datasets' / 'bruno'
    output_filename = Path(__file__).parent.parent / 'outputs' / 'bruno.csv'
    
    # Ensure the output directory exists before saving
    os.makedirs(output_filename.parent, exist_ok=True)
    
    process_bruno_dataset(dataset_dir, output_filename)