
# Accelerometer Dataset Combiner

This tool processes multiple accelerometer sources and outputs one consolidated CSV file per distinct dataset, keeping separate experimental datasets independent. These cleaned files are ready for absorption into machine learning models.

---

## Datasets

### [Luo Dataset](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463)

* **Surfaces:** `'calibration'`, `'cobble_stone'`, `'stairs_up'`, `'stairs_down'`, `'slope_up'`, `'slope_down'`, `'flat'`, `'bank_left'`, `'bank_right'`, `'grass'`
* **Base Columns:** `time_ms`, `sid`, `surface`, `trial_no`
* **Sensor Naming Convention:** `{side}_{location}_{Metric}_{axis}` (e.g., `left_thigh_Acc_x`)
* **Sensor Locations:** `left_thigh`, `right_thigh`, `right_wrist`, `right_shank`
* **Metrics Tracked:** `SampleTimeFine`, `Acc`, `FreeAcc`, `Gyr`, `Mag`, `VelInc`, `OriInc` (q0-q3), `Roll`, `Pitch`, `Yaw`, `Latitude`, `Longitude`, `Altitude`, `Vel`

### Beach Dataset

* **Surfaces:** `'flat'`
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** `left_wrist`, `right_wrist`, `left_upper_leg`, `right_upper_leg`, `left_lower_leg`, `right_lower_leg`, `left_foot`, `right_foot`

### Bruno Dataset

* **Surfaces:** `'brush_teeth'` , `'stairs_up'` (mapped from `climb_stairs` ), `'comb_hair'` , `'stairs_down'` (mapped from `descend_stairs` ), `'drink_glass'` , `'eat_meat'` , `'eat_soup'` , `'getup_bed'` , `'liedown_bed'` , `'pour_water'` , `'sitdown_chair'` , `'standup_chair'` , `'use_telephone'` , `'flat'` (mapped from `walk` )
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** Right Wrist 
* **Metrics Tracked:** Acceleration in x, y, and z axes (converted to m/s^2) 
* **Note:** This dataset contains strictly recorded, experimental accelerometer data; no generated data is present. The `_MODEL` directories have been intentionally omitted to ensure duplicate readings are not merged into the independent dataset [cite: 1].

### Karas Dataset

* **Surfaces:** `'non-study activity'`, `'clapping'`, `'flat'`, `'stairs_up'`, `'stairs_down'`, `'driving'`
* **Base Columns:** `time_ms`, `sid`, `surface`
* **Sensor Locations (Acceleration x/y/z):** `left_wrist`, `left_hip`, `left_ankle`, `right_ankle`

### HuGaDB Dataset (Version 1 & Version 2)

*Note: Version 1 and Version 2 are distinct datasets and are processed into entirely separate CSV files to maintain their independence.*

* **Surfaces:** `'flat'`, `'running'`, `'stairs_up'`, `'stairs_down'`, `'sitting'`, `'sitting_down'`, `'standing_up'`, `'standing'`, `'bicycling'`, `'elevator_up'`, `'elevator_down'`, `'driving'`
* **Base Columns:** `time_ms`, `sid`, `surface`, `trial_no`
* **Sensor Naming Convention:** `{side}_{location}_{Metric}_{axis}` (e.g., `right_foot_Acc_x`)
* **Sensor Locations:** `right_foot`, `left_foot`, `right_shank`, `left_shank`, `right_thigh`, `left_thigh`
* **Metrics Tracked:** `Acc` (Accelerometer), `Gyr` (Gyroscope), `EMG` (Electromyography)

---

## Time Column
