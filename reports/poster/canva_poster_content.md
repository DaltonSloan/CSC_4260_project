# Canva Poster Copy

## Title
Passive Occupancy Estimation in Smart Classrooms Using CO2, VOC, Airflow, and Indoor Air Quality Signals

## Authors
Fengjun Han, Samuel Hartmann, Dalton Sloan, Garrett Green

## Affiliation
Tennessee Technological University, Department of Computer Science / Ashraf Islam Engineering Building

## Intro / Significance / Research Question
Modern HVAC systems often follow fixed schedules instead of actual room usage. That wastes energy during low-occupancy periods and misses opportunities for occupancy-aware control. This project asks whether passive building signals already collected by the KODE smart-building platform in the Ashraf Islam Engineering Building (AIEB) can estimate occupancy without cameras or badge scanners.

Research question:
- Can CO2, VOC, humidity, temperature, and airflow be used to estimate room occupancy continuously?
- Can a physics-based CO2 mass-balance formula provide useful occupancy estimates without labeled training data?

Why it matters:
- Supports occupancy-aware HVAC scheduling
- Reduces unnecessary runtime during low-use periods
- Avoids privacy concerns of camera-based counting

## Dataset
Data source: KODE smart-building platform — point-history exports for IAQ and fan-powered box (FPB) systems in the AIEB building.

Room 354 was the primary analysis room. We used a 30-day IAQ + FPB export from 2026-02-27 to 2026-03-29, resampled to 5-minute intervals for about 8,600 time steps. Available signals included CO2 (ppm), VOC, humidity (%), temperature (F), and discharge airflow (cfm).

Room 361 was used as a limited validation case. We used FPB exports from April 3–9, 2026 plus 3 manual headcount anchors taken directly in the classroom.

Class enrollment records were parsed into schedule windows to provide occupancy context. Room 354 enrollment data confirmed a nominal capacity of roughly 35 students.

Data processing:
- Filter relevant point-history records from KODE exports
- Pivot long-format exports into room-level columns
- Resample to 5-minute intervals
- Merge IAQ and FPB streams on time index
- Convert discharge airflow to ACH using room volume (50 ft × 30 ft × 15 ft ≈ 637.1 m³)
- Retain missing humidity and temperature values rather than imputing them

## Methods
- Physics-based CO2 mass-balance formula to estimate occupancy from CO2 buildup, room volume, and ventilation rate
- Blended occupancy estimate: 70% CO2 anchor + 30% VOC/humidity/temperature index
- Correlation analysis across all IAQ signals to identify the strongest occupancy indicators
- Exploratory visualization: 4-panel time-series comparison, correlation matrix, occupancy curve with airflow overlay
- Manual headcount anchors (Room 361) used to qualitatively assess estimate plausibility

Key assumptions:
- Room dimensions: 50 ft × 30 ft × 15 ft (~637.1 m³)
- Outdoor CO2 baseline: ~416 ppm (measured ambient)
- Per-person CO2 generation: 0.018 m³/h/person (light activity)
- Measured airflow treated as a ventilation proxy; converted to ACH

CO2 sensor lag: CO2 builds up gradually near the sensor depending on room volume, ventilation rate, and occupancy density. Estimated occupancy peaks can lag actual room occupancy by up to ~1 hour, especially early in a class period when people have just arrived.

Vibration signal: Vibration activity from the building's structural sensors was observed to spike at the start, end, and between class periods — consistent with students moving between rooms. This pattern supports identification of occupied vs. unoccupied windows.

## Results
Correlation findings (Room 354, 30-day dataset):
- CO2 ↔ estimated occupancy: r = 0.978
- VOC ↔ estimated occupancy: r = 0.748
- CO2 ↔ VOC: r = 0.751
- Temperature ↔ CO2: r = 0.414 (positive linear relationship)
- Humidity ↔ VOC: r = 0.40 (positive linear relationship)
- Airflow ↔ CO2: r = -0.064 (weak; airflow dilutes rather than drives CO2)

Room 354 physics estimate summary:
- Mean estimated occupancy: 4.22 people
- P90: 13.38 people
- P99: 29.78 people
- Max estimate: 46.0 people

Effect of measured airflow vs. fixed 4 ACH assumption:
- Fixed 4 ACH baseline mean: 5.22 people
- Measured-airflow baseline mean: 4.17 people (~20% reduction)
- Peak estimate reduced by 35%
- Intervals above 30 people dropped from 150 to 85 (-43%)
- Occupancy-curve volatility reduced by 16%

Room 361 qualitative reference check (3 manual headcounts):
- Manual counts: 33 people (Apr 7 ~14:38), 11 people (Apr 8 ~15:00), 13 people (Apr 8 ~15:30)
- Physics formula correctly estimated the lower-occupancy periods (errors under 1.5 people for the 11 and 13 person counts)
- Largest miss: 33-person period with high ventilation (~1,130 cfm) — active airflow diluted CO2 faster than occupants could build it up, illustrating the CO2 lag limitation

## Discussion
CO2 and VOC were the clearest passive occupancy signals in this classroom setting, with temperature and humidity showing a positive linear relationship to both but providing weaker standalone occupancy discrimination. Airflow was not a direct headcount proxy — instead, it improved the estimate by adjusting how quickly ventilation diluted indoor CO2.

The CO2 sensor lag is a key physical constraint: because CO2 accumulates gradually in a room of this volume (637 m³), the estimated occupancy curve can lag actual room use by up to one hour. In practical terms, the physics formula is more reliable for detecting sustained occupancy than instantaneous headcount.

Vibration data from class transition periods (start, end, between sessions) aligned with expected student movement patterns, supporting that the physical environment responds to occupancy-driven activity.

The current analysis is exploratory. No machine learning models have been trained yet. The physics-based CO2 mass-balance formula is the sole estimation method; machine learning is planned as future work once sufficient labeled data are available.

## Conclusions / Recommendations
- CO2 and VOC are the strongest passive occupancy indicators; temperature and humidity have a positive linear relationship with both but provide secondary value
- Measured airflow improves occupancy estimates by replacing an unrealistic fixed-ventilation assumption — reducing mean estimate by ~20% and peak spikes by 35%
- Vibration signals at class transitions provide supplemental evidence of occupancy-driven activity
- The CO2 sensor lag (up to ~1 hour) means the physics estimate tracks sustained occupancy better than instantaneous headcount
- The physics-based formula provides a plausible continuous occupancy estimate without any labeled training data

Future work:
- Collect labeled occupancy data in Rooms 354 and 361 to enable supervised modeling
- Integrate TDMS vibration data aligned to the 5-minute room timeline
- Train an LSTM temporal model on the Jetstream HPC cluster once labeled data and vibration signals are available
- Add outdoor-air fraction or damper-position data to improve ventilation accuracy
- Evaluate coarse occupancy classes (low / medium / high) for HVAC control deployment

## References
1. ASHRAE 62.1. Ventilation for Acceptable Indoor Air Quality.
2. Fisk, W. J., et al. CO2-based occupancy estimation in commercial buildings.
3. Tennessee Tech AIEB FPB and IAQ point-history exports via KODE platform, 2026.
4. KODE smart-building platform, Tennessee Tech AIEB building sensor network.

## Acknowledgements
- Domain experts: Chandler Norman, Norman Walker, Elisabeth Humphrey, Dr. Steven Anton
- Tennessee Tech University and the AIEB building KODE data platform
- CSC 4260 project support team
- Jetstream HPC cluster (planned for future LSTM modeling)

## Suggested Figure Placement
- Main center figure: `reports/figures/room354_estimated_occupancy.png`
- Supporting figure: `reports/figures/room354_feature_comparison.png`
- Correlation matrix: `reports/figures/room354_feature_comparison.png`
- Validation figure: Room 361 anchor-window evaluation table from `reports/room361_pipeline/best_anchor_window_evaluation.md`
