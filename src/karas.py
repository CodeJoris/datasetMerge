import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures

g = 9.80665  # Acceleration due to gravity in m/s^2

def get_id(file_path: Path) -> str:
    '''
    Takes in the file path and returns the file name 
    which corresponds to the subject id
    '''
    return file_path.stem

def get_surface(activity_label: pd.Series) -> pd.Series:
    '''
    Takes the activity_label column from the csv file and
    returns a column of identical shape with the corresponding 
    surface
    '''
    label_mapping = {
        1: "flat",
        2: "stairs_up",
        3: "stairs_down",
        4: "driving",
        77: "clapping",
        99: "non-study activity"
    }
    # map the values according to the dictionary
    return activity_label.map(label_mapping)

def process_subject(file_name: Path) -> pd.DataFrame:
    '''
    Takes in a file_name and returns the dataframe of the 
    subject with the activity column replaced by the 
    surface column mapped by the get_surface function
    '''
    # Read the data file
    df = pd.read_csv(file_name)
    
    # Extract ID and generate surface mappings
    df['sid'] = get_id(file_name)
    df['surface'] = get_surface(df['activity'])
    
    # Drop original activity column and rename time column
    df = df.drop(columns=['activity'])
    df = df.rename(columns={'time_s': 'time_ms'})
    df['time_ms'] = (df['time_ms'] * 1000).astype(int)  # Convert seconds to milliseconds
    
    
    # Reorder columns to place time, subject_id, and surface first
    accel_cols = [col for col in df.columns if col not in ['time_ms', 'sid', 'surface']]
    df[accel_cols] = df[accel_cols] * g  # Convert acceleration from g to m/s^2
    ordered_cols = ['time_ms', 'sid', 'surface'] + accel_cols
    
    return df[ordered_cols]

def main(file_paths: list[Path], output_path: Path):
    '''
    Merges all subject data together into one dataframe and 
    exports it as a csv to the output_path. Using multiprocessing
    to speed up the process.
    
    The output csv should have the following columns:
    time_ms, subject_id, surface, all acceleration data
    '''    
    dataframes = []
    # Use multiprocessing to parse files concurrently[cite: 1]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_subject, file_paths)
        dataframes = list(results)
    
    if dataframes:
        # Merge into a single dataframe
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write to csv[cite: 1]
        out_file = output_path / 'karas.csv'
        merged_df.to_csv(out_file, index=False)
        print(f"Data successfully merged and saved to {out_file}")

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / 'datasets' / 'karas1' / 'data'
    OUTPUT_PATH = Path(__file__).parent.parent / 'outputs' 
    files = list(DATA_PATH.rglob("*.csv"))

    main(files, OUTPUT_PATH)