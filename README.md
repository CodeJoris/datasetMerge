# Accelerometer Dataset Combiner

This tool processes multiple accelerometer sources and outputs one consolidated CSV file per distinct dataset, keeping separate experimental datasets independent. These cleaned files are ready for absorption into machine learning models.

---

## Datasets

### [Luo Dataset](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463)

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
* **Note:** This dataset contains strictly recorded, experimental accelerometer data; no generated data is present. The `_MODEL` directories have been intentionally omitted to ensure duplicate readings are not merged into the independent dataset [cite: 1].

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
