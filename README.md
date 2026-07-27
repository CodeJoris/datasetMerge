# Accelerometer Dataset Combiner

This tool processes multiple accelerometer sources and outputs one consolidated CSV file per distinct dataset, keeping separate experimental datasets independent. These cleaned files are ready for absorption into machine learning models.

---

## Output File Example

Each dataset is written to `outputs/<dataset>.csv` with the shared base columns first, followed by the sensor measurement columns.

The column order is:

* With trials: `time_ms, sid, surface, trial_no, ...sensor columns...`
* Without trials: `time_ms, sid, surface, ...sensor columns...`

Example layout:

```csv
time_ms,sid,surface,trial_no,left_thigh_Acc_x,left_thigh_Acc_y,left_thigh_Acc_z,...
0,1,stairs_up,1,...
10,1,stairs_up,1,...
```

For datasets that do not track trials, the `trial_no` column is omitted and the file starts with `time_ms,sid,surface,...`.

Base column meanings:

* **`time_ms`**: elapsed time in milliseconds from the start of the subject recording or trial. Values should be numeric and increase as the recording progresses. *Note that constant time increase per trial was enforced, meaning if the source data contained missing packets, the output will fill those rows with NaN (represented as ,, in the csv).*
* **`sid`**: integer subject or session identifier assigned by the loader. This is used to keep data from different people or sessions separate.
* **`surface`**: standardized activity label for the recording, using the dataset-specific mapping shown below.
* **`trial_no`**: integer trial or file number. This appears only when the source dataset is naturally split into multiple trials for the same subject and activity. In this project, that applies to the Luo and HuGaDB outputs.
* **`data columns`**: Accelerometer and gyroscope data are converted to `m/s^2` and `rads/s` respectively

## Datasets

### Luo Dataset

* **Surfaces:** `'calibration'`, `'cobble_stone'`, `'stairs_up'`, `'stairs_down'`, `'slope_up'`, `'slope_down'`, `'flat'`, `'bank_left'`, `'bank_right'`, `'grass'`
* **Base Columns:** `time_ms`, `sid`, `surface`, `trial_no`
* **Sensor Naming Convention:** `{side}_{location}_{Metric}_{axis}` (e.g., `left_thigh_Acc_x`)
* **Sensor Locations:** `trunk`, `right_wrist`, `right_thigh`, `left_thigh`, `right_shank`, `left_shank`
* **Metrics Tracked:** `sampletimefine`, `acc`, `freeacc`, `gyr`, `mag`, `velinc`, `oriinc`, `roll`, `pitch`, `yaw`, `latitude`, `longitude`, `altitude`, `vel`

### Beach Dataset

* **Surfaces:** `'flat'`
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** `right_foot`, `left_wrist`, `left_foot`, `left_lower`, `right_wrist`, `right_lower`, `left_upper`, `right_upper`

### Bruno Dataset

* **Surfaces:** `'brush_teeth'` , `'stairs_up'` (mapped from `climb_stairs` ), `'comb_hair'` , `'stairs_down'` (mapped from `descend_stairs` ), `'drink_glass'` , `'eat_meat'` , `'eat_soup'` , `'getup_bed'` , `'liedown_bed'` , `'pour_water'` , `'sitdown_chair'` , `'standup_chair'` , `'use_telephone'` , `'flat'` (mapped from `walk` )
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** `right_wrist`
* **Metrics Tracked:** `acc`

### Karas Dataset

* **Surfaces:** `'non-study activity'`, `'clapping'`, `'flat'`, `'stairs_up'`, `'stairs_down'`, `'driving'`
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** `left_hip`, `left_wrist`, `right_ankle`, `left_ankle`

### HuGaDB Dataset (Version 2)

*Note: Version 2 is the same as Version 1 just relabeled.*

* **Surfaces:** `'flat'`, `'running'`, `'stairs_up'`, `'stairs_down'`, `'sitting'`, `'sitting_down'`, `'standing_up'`, `'standing'`, `'bicycling'`, `'elevator_up'`, `'elevator_down'`, `'driving'`
* **Base Columns:** `time_ms`, `sid`, `surface`, `trial_no`
* **Sensor Naming Convention:** `{side}_{location}_{Metric}_{axis}` (e.g., `right_foot_Acc_x`)
* **Sensor Locations:** `right_foot`, `left_foot`, `right_shank`, `left_shank`, `right_thigh`, `left_thigh`
* **Metrics Tracked:** `acc` (accelerometer), `gyr` (gyroscope), `emg` (electromyography)

### mHealth Dataset

* **Surfaces:** `'non-study activity'`, `'standing'`, `'sitting'`, `'lying_down'`, `'flat'`, `'stairs_up'`, `'waist_bends_forward'`, `'frontal_elevation_arms'`, `'knees_bending'`, `'bicycling'`, `'jogging'`, `'running'`, `'jump_front_back'`
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations:** `chest`, `left_ankle`, `right_lower`
* **Metrics Tracked:** `acc` (accelerometer), `gyr` (gyroscope), `mag` (magnetometer), `ecg` (electrocardiogram)

---

## Project File Structure

```
datasetMerge/
├── datasets/                          # Raw downloaded datasets (directory per dataset)
│   ├── luo/                          # Luo dataset files
│   ├── karas/                        # Karas dataset files
│   ├── bruno/                        # Bruno dataset files
│   ├── beach/                        # Beach dataset files
│   ├── hugadb/                       # HuGaDB dataset files
│   └── mhealth/                      # mHealth dataset files
├── dataloading/                       # Dataset loading module
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── all_datasets.py
│   └── [dataset_name].py             # Individual dataset loaders
├── src/                               # Dataset processing scripts
│   ├── [dataset_name].py             # Processing scripts for each dataset
│   └── all_datasets.py
├── outputs/                           # Final merged CSV files
│   ├── luo.csv
│   ├── karas.csv
│   ├── bruno.csv
│   ├── beach.csv
│   ├── hugadb.csv
│   └── mhealth.csv
├── tests/                             # Test suite
│   ├── run_tests.py
│   └── [test_name]_test.py
├── datasets.yaml                      # Configuration file with dataset metadata
└── README.md                          # This file
```

---

## Getting Started: Download & Setup

### Step 1: Download Datasets

Each dataset must be downloaded from the provided links and placed in the `datasets/` directory. Here's where to download each one:

#### Luo Dataset

- **[Download URL:](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463)**
- **Save to:** `datasets/luo/`
- **Notes:** Extract all files to this directory. Should contain participant folders (1, 2, 3, etc.)

#### Karas Dataset

- **[Download URL:](https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/)**
- **Save to:** `datasets/karas/`
- **Notes:** Create the `karas/data/` directory and place all data `.csv` files in it

#### Bruno Dataset

- **[Download Link](https://archive.ics.uci.edu/dataset/283/dataset+for+adl+recognition+with+wrist+worn+accelerometer)**
- **Save to:** `datasets/bruno/`
- **Notes:** Have the activity sub-directories inside this directory eg: `bruno/Brush_teeth/` or `bruno/Climb_stairs/`

#### Beach Dataset

- **[Download Link
  ](https://data.mendeley.com/datasets/5rrxw7y5bj/1)**
- **Save to:** `datasets/beach/`
- **Notes:** Have all .pkl files directly in `beach/` directory

#### HuGaDB Dataset (v2)

- **[Download Link](https://www.dropbox.com/scl/fi/jsq0mr26rthrzy64vkkjc/HuGaDB-v2.zip?rlkey=101j8lvdktdejm105cf9fpisi&e=1&dl=0)**
- **Save to:** `datasets/hugadb/`
- **Notes:** Extract all `.txt` data files should be stored in `hugadb/v2/`

#### mHealth Dataset

- **[Download URL:](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)**
- **Save to:** `datasets/mhealth/`
- **Notes:** Extract all data (.log files) directly to this directory.

### Step 2: Verify Directory Names

The data loaders expect specific directory names in `datasets/`. If you've downloaded with different names, rename them as follows:


| Expected Directory | Common Download Name                                                                          | Action                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `luo`              | `input data_SD.zip`                                                                           | Extract and rename to`luo`                                                     |
| `karas`            | `labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0.zip` | Extract rename the dir to`karas` and rename `raw_accelerometry_data` to `data` |
| `bruno`            | `dataset+for+adl+recognition+with+wrist+worn+accelerometer.zip`                               | Extract and rename to`bruno`                                                   |
| `beach`            | `5rrxw7y5bj-1.zip`                                                                            | Extract and rename to`beach`                                                   |
| `hugadb`           | `HuGaDB-v2.zip`                                                                               | Extract and rename folder to`hugadb`                                           |
| `mhealth`          | `dataset+for+adl+recognition+with+wrist+worn+accelerometer.zip`                               | Rename to`mhealth`                                                             |

### Step 3: Configure datasets.yaml (Optional)

The `datasets.yaml` file contains metadata for each dataset including:

- Download links
- Sampling frequencies (fs)
- Sensor locations
- Metrics tracked
- Unit conversions

If you need to modify paths or add custom datasets, edit this file accordingly. Most users can leave this as-is.

### Step 4: Run Processing

Once all datasets are downloaded and properly named:

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate  # On Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # On Windows PowerShell

pip install -r requirements.txt

# Run the dataset processing
python src/[dataset_name].py
```

Processed datasets will be saved to `outputs/` as individual CSV files (e.g., `outputs/luo.csv`, `outputs/karas.csv`, etc.).
