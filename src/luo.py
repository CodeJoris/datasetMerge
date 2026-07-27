import os, glob
import pandas as pd
import concurrent.futures
from pathlib import Path
import shutil

'''
fs = 100Hz

Acc_X : in the vertical direction (w/gravity)
Acc_Y : in the medio-lateral direction (w/gravity)
Acc_Z : in the anterior-posterior direction (w/gravity)
'''

DATA_PATH = Path(__file__).parent.parent / "datasets" / "luo"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"


def main(data_path: Path = DATA_PATH, output_path: Path = OUTPUT_PATH) -> None:
    '''
    Orchestrates the multiprocessing pipeline to read, clean, and concatenate 
    subject sensor data into a single CSV.

    Parameters:
    ---
    data_path: Path
        The root directory path containing the subject data folders.
    output_path: Path
        The directory path where the final concatenated CSV will be saved.

    Returns: None
    '''
    file_paths = get_filepaths(data_path)
    all_subjects_data = []

    # Initialize multiprocessing pool
    print("Starting multiprocessing execution...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the process_subject function to all subjects
        futures = {executor.submit(process_subject, sid, paths): sid for sid, paths in file_paths.items()}

        # Collect results as they finish processing
        for future in concurrent.futures.as_completed(futures):
            try:
                result_df = future.result()
                if not result_df.empty:
                    all_subjects_data.append(result_df)
            except Exception as exc:
                sid = futures[future]
                print(f'Subject {sid} generated an exception: {exc}')

    if all_subjects_data:
        print("Concatenating final dataset...")
        final_df = pd.concat(all_subjects_data, ignore_index=True)

        time_series = final_df.groupby(['sid', 'trial_no'])['packetcounter'].transform(lambda x: (x - x.min()) * 10)
        final_df.insert(0, 'time_ms', time_series)
        final_df.columns = [str(column).lower() for column in final_df.columns]

        # Ensure output directory exists before saving
        output_path.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(output_path / "luo.csv", index=False, na_rep='')
        print(f"Successfully saved concatenated dataset to {output_path / 'luo.csv'}")
    else:
        print("No data was processed.")

def process_subject(sid: int, paths: list[Path]) -> pd.DataFrame:
    '''
    Worker function that processes and merges all sensor trials for a single subject.

    Parameters:
    ---
    sid: int
        The subject's numerical ID.
    paths: list[Path]
        A list of file paths to all CSV files associated with the subject.

    Returns: pd.DataFrame
    ---
        A single dataframe containing all merged trial data for the subject.
    '''
    subject_trials_data = []
    
    # 1. Group the files
    trials = group_paths_by_trial(paths)

    for trial_no, trial_paths in trials.items():
        surface = get_surface(trial_paths[0].name)
        
        # 2. Process all sensor files for this trial cleanly using a list comprehension
        dfs_to_concat = [process_sensor_file(path) for path in trial_paths]

        if dfs_to_concat:
            # 3. Merge the trial sensors with an outer join to retain all recorded packets
            merged_df = pd.concat(dfs_to_concat, axis=1, join="outer")
            
            # De-fragment IMMEDIATELY after concatenation before doing anything else
            merged_df = merged_df.copy()

            # Enforce linear time increase by reindexing over the full packet range
            min_packet = merged_df.index.min()
            max_packet = merged_df.index.max()
            full_packet_range = pd.Index(range(min_packet, max_packet + 1), name="packetcounter")
            
            # This fills missing packets in the sequence with NaNs
            merged_df = merged_df.reindex(full_packet_range).reset_index()

            # 4. Append metadata
            merged_df.insert(1, "sid", sid)
            merged_df.insert(2, "surface", surface)
            merged_df.insert(3, "trial_no", trial_no)
            
            subject_trials_data.append(merged_df)

    if subject_trials_data:
        return pd.concat(subject_trials_data, ignore_index=True)
    return pd.DataFrame()

def group_paths_by_trial(paths: list[Path]) -> dict[str, list[Path]]:
    '''
    Groups a list of file paths based on their trial number prefix.

    Parameters:
    ---
    paths: list[Path]
        A list of file paths to be grouped.

    Returns: dict[str, list[Path]]
    ---
        A dictionary where keys are the trial number strings and values 
        are lists of file paths belonging to that trial.
    '''
    trials = {}
    for path in paths:
        trial_no = path.name.split('-')[0]
        if trial_no not in trials:
            trials[trial_no] = []
        trials[trial_no].append(path)
    return trials

def process_sensor_file(path: Path) -> pd.DataFrame:
    '''
    Reads a single sensor CSV file, strips whitespace from column names, 
    removes the "SampleTimeFine" column, renames remaining columns with 
    the sensor location prefix, and sets "PacketCounter" as the index.

    Parameters
    ---
    path: Path
        The file path to the sensor's CSV file.

    Returns
    ---
    pd.DataFrame
        A cleaned dataframe indexed by PacketCounter, ready for merging.
    '''
    df = pd.read_csv(path, engine='python', on_bad_lines='skip', skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    loc = get_location(path.name)

    if "sampletimefine" in df.columns:
        df = df.drop(columns=["sampletimefine"])

    rename_mapping = {
        col: f"{loc}_{col}" for col in df.columns if col != "packetcounter"
    }
    df = df.rename(columns=rename_mapping)
    
    return df.set_index("packetcounter")

def get_sid(file_path: Path) -> int:
    '''
    Extracts the subject ID from the parent directory of a given file path.

    Parameters:
    ---
    file_path: Path
        The file path to evaluate.

    Returns: int
    ---
        The subject ID extracted from the directory structure.
    '''
    return int(file_path.parts[-2])

def get_surface(file_name: str) -> str:
    '''
    Takes a surface number (first digit before the - in the filename)
    and returns the corresponding surface

    Pararmeters:
    ---
    surface_no: int
        The number before the - in the filename.


    Returns: str
    ---
        surface based on file number as input
    '''
    surface_no = int(file_name.split('-')[0])

    surface_ref = {
        "calibration":[1,2,3],
        "flat":[4,5,6,7,8,9],
        "cobble_stone":[10,11,12,13,14,15],
        "stairs_up":[16,18,20,22,24,26],
        "stairs_down":[17,19,21,23,25,27],
        "slope_up":[28,30,32,34,36,38],
        "slope_down":[29,31,33,35,37,39],
        "bank_left":[40,42,44,46,48,50],
        "bank_right":[41,43,45,47,49,51],
        "grass":[52,53,54,55,56,57]
    }
    for surface, numbers in surface_ref.items():
        if surface_no in numbers:
            return surface
    return 'unknown'

def get_location(file_name: str) -> str:
    '''
    Takes in a file_name, extracts the last 2 characters
    that represent the sensor location and return 
    a string of the sensor location (trunk, etc...)

    Parameters:
    ---
    file_name: str
        name of the file being processed

    Returns: str
    ---
        Location of the sensor based on the file name
    '''
    location_id = file_name.split('.')[0][-2:]

    location_ref = {
        "CC":"trunk",
        "9B":"right_shank",
        "B6":"left_shank",
        "95":"right_wrist",
        "93":"right_thigh", # right thigh
        "8B":"left_thigh" # left thigh
    }
    return location_ref[location_id]

def get_filepaths(data_path: Path) -> dict[int, list[Path]]:
    '''
    Returns: dict[int, list[str]]
    ---
        A dict with keys the subject id and values
        a list of the files
    '''
    filepaths = {}
    for subject_id in range(1, 31):
        subject_dir = data_path / f'{subject_id}'
        filepaths[subject_id] = list(subject_dir.rglob("*.csv"))
    
    return filepaths

def append_missing_subjects(missing_sids: list[int], data_path: Path, output_file: Path) -> None:
    '''Processes specific subjects and appends them to an existing CSV file safely.'''
    
    # 1. Read ONLY the header of the existing file to get the exact column layout
    existing_columns = pd.read_csv(output_file, nrows=0).columns
    
    for sid in missing_sids:
        print(f"Processing missing Subject {sid}...")
        subject_dir = data_path / str(sid)
        paths = list(subject_dir.rglob("*.csv"))
        
        if not paths:
            print(f"No files found for Subject {sid}.")
            continue
            
        subject_df = process_subject(sid, paths)
        
        if not subject_df.empty:
            print(f"Appending Subject {sid} to {output_file.name}...")
            
            # 2. Reindex the dataframe to match the target CSV exactly.
            # This fills missing columns with NaN (empty strings in CSV) and drops extra columns.
            subject_df = subject_df.reindex(columns=existing_columns)
            
            # 3. Now it is safe to append
            subject_df.to_csv(output_file, mode='a', index=False, header=False, na_rep='')
            print(f"Subject {sid} successfully added!")

def rename_csv_column(input_file: Path, output_file: Path, old_name: str, new_name: str) -> None:
    '''
    Streams a massive CSV file to a new file, renaming a specific column 
    in the header while keeping memory usage near zero.
    '''
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # 1. Read just the first line (the header)
        header_line = infile.readline().strip()
        
        # 2. Split it safely to prevent renaming partial matches 
        # (e.g., we don't want changing "Acc_X" to affect "left_thigh_Acc_X")
        columns = header_line.split(',')
        
        # 3. Find the exact column and rename it
        updated_columns = [new_name if col == old_name else col for col in columns]
        
        # 4. Write the new header to the output file
        outfile.write(','.join(updated_columns) + '\n')
        
        # 5. Blast the rest of the experimental data directly to the new file
        # This streams in chunks at the I/O level, bypassing Python's line parser
        shutil.copyfileobj(infile, outfile)
        
    print(f"Successfully renamed '{old_name}' to '{new_name}' in {output_file.name}")

if __name__ == "__main__":
    # Specify the missing subject ID (or a list of them if multiple failed)
    missing_subjects = []
    target_csv = OUTPUT_PATH / "luo.csv"

    # Example usage:
    input_path = target_csv
    output_path = OUTPUT_PATH / 'luo_renamed.csv'

    if len(missing_subjects) == 0:
        main()
    else:
        append_missing_subjects(missing_subjects, DATA_PATH, target_csv)