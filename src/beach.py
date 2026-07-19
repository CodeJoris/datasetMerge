import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures

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
    # Since all trials are on treadmills, we return a series of 'flat'
    return pd.Series('flat', index=activity_label.index)

def process_subject(file_path: Path) -> pd.DataFrame:
    '''
    Takes in a file_path (.pkl) and returns the data in 
    the form of a pd.Dataframe ready to be appended to the
    combined df.
    '''
    subject_id = get_id(file_path)
    
    with open(file_path, 'rb') as pkl_file:
        data = pd.read_pickle(pkl_file)
        
    sensors = [
        'left_wrist', 'right_wrist', 
        'left_upper_leg', 'right_upper_leg', 
        'left_lower_leg', 'right_lower_leg', 
        'left_foot', 'right_foot'
    ]
    
    merged_df = None
    
    for sensor in sensors:
        if sensor in data and isinstance(data[sensor], pd.DataFrame):
            sensor_df = data[sensor].copy()
            
            sensor_df['DateTime'] = sensor_df['DateTime'].dt.round('10ms')
            sensor_df = sensor_df.groupby('DateTime', as_index=False).mean()
            
            sensor_df = sensor_df.rename(columns={
                'x': f'{sensor}_Acc_x',
                'y': f'{sensor}_Acc_y',
                'z': f'{sensor}_Acc_z'
            })
            
            if merged_df is None:
                merged_df = sensor_df
            else:
                merged_df = pd.merge(merged_df, sensor_df, on='DateTime', how='outer')
    
    if merged_df is not None:
        merged_df = merged_df.sort_values('DateTime').reset_index(drop=True)
        
        # Isolate the acceleration columns for targeted interpolation
        accel_cols = [c for c in merged_df.columns if c != 'DateTime']
        
        # TARGETED TRANSFORMATION: Interpolate internal data gaps without extending edge padding
        merged_df[accel_cols] = merged_df[accel_cols].interpolate(method='linear', limit_area='inside')
        
        # OPTIONAL TRANSFORMATION: Drop terminal rows where no sensors were active
        merged_df = merged_df.dropna(subset=accel_cols, how='any')
        
        merged_df = merged_df.rename(columns={'DateTime': 'time_ms'})
        min_time = merged_df['time_ms'].min()
        merged_df['time_ms'] = (merged_df['time_ms'] - min_time).dt.total_seconds() * 1000
        merged_df['time_ms'] = merged_df['time_ms'].astype(int)
        
        merged_df['sid'] = subject_id
        merged_df['surface'] = get_surface(merged_df['sid'])
        
        metadata_cols = ['time_ms', 'sid', 'surface']
        merged_df = merged_df[metadata_cols + accel_cols]
        
    return merged_df

def main(file_paths: list[Path], output_path: Path):
    '''
    Merges all subject data together into one dataframe and 
    exports it as a csv to the output_path. Using multiprocessing
    to speed up the process.
    
    The output csv should have the following columns:
    time, subject_id, surface, all acceleration data
    '''    
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / 'beach.csv'
    
    dfs = []
    
    # Execute subject processing in parallel
    print(f"Processing {len(file_paths)} subject files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_subject, file_paths)
        
        for df in results:
            if df is not None:
                dfs.append(df)
                
    if dfs:
        print("Concatenating all subject dataframes...")
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Reorder columns: time, subject_id, surface, [sensor data...]
        metadata_cols = ['time_ms', 'sid', 'surface']
        accel_cols = [c for c in final_df.columns if c not in metadata_cols]
        final_df = final_df[metadata_cols + accel_cols]
        
        print(f"Exporting to {out_file}...")
        final_df.to_csv(out_file, index=False)
        print("Export complete!")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / 'datasets' / 'beach1'
    OUTPUT_PATH = Path(__file__).parent.parent / 'outputs' 
    files = list(DATA_PATH.rglob("*.pkl"))

    if not files:
        print(f"No .pkl files found in {DATA_PATH}")
    else:
        main(files, OUTPUT_PATH)