# Room 354 Occupancy Estimation Using VOC, CO2, Humidity, and Temperature

Team Members: Samuel Hartmann, Fengjun Han, Garrett Green, Dalton Sloan

Domain Experts: Chandler Norman, Norman Walker, Elisabeth Humphrey, Dr. Steven Anton

## 1. Problem Statement

The original project plan focused on evaluating HVAC efficiency in the Ashraf Islam Engineering Building at Tennessee Tech. After exploratory analysis, the project focus was narrowed to Room 354 occupancy estimation using indoor air quality signals. The current notebook-based question is whether occupancy can be inferred from two Room 354 sensor streams containing VOC, CO2, humidity, and temperature, with vibration data planned as a future confirmation signal.

This occupancy objective matters because occupancy-aware controls can reduce wasted HVAC runtime while still maintaining indoor air quality. For the current notebook phase, success is defined informally as: strong and interpretable co-movement among occupancy-related signals, a plausible per-timestamp occupancy estimate, and a model structure that can later be checked against vibration and airflow data once those sources are integrated.

## 2. Background and Related Work

This section is not required for CSC 4260, so the current report focuses on the direct data, methods, and findings from the Room 354 occupancy notebook.

TODO: Add a short background summary of occupancy detection using indoor air quality and activity signals only if the team decides it is needed for the final submission.

## 3. Data and Exploratory Analysis

The repository currently contains multiple building and room-level HVAC datasets, including:

- whole-building energy data,
- Room 354 data with carbon dioxide, VOC, noise level, and zone temperature fields,
- Room 361 data with carbon dioxide, humidity, and zone temperature fields,
- additional 361 HVAC data with discharge air temperature, zone temperature, zone carbon dioxide, airflow-related fields, and effective occupancy fields.

Work already completed in the repository includes:

- collecting and organizing the available CSV files,
- preparing a workflow for loading HVAC data into a database,
- creating an initial notebook that plots whole-building energy consumption over time,
- creating an initial whole-building plot with a 24-hour moving average,
- creating an initial notebook that plots discharge air temperature and zone temperature for 361 data,
- creating a Room 354 notebook that merges two sensors and compares VOC, CO2, humidity, and temperature over time,
- generating a cleaned merged Room 354 dataset for analysis (`reports/room354_multisensor_merged.csv`).

Occupancy-focused exploratory plots currently supported by the Room 354 notebook include:

- a 3-panel time-series comparison of VOC/CO2, humidity, and temperature,
- a separate estimated occupancy time series derived from the blended Room 354 model.

TODO: Add notes from any domain expert conversations that have already happened.

The Room 354 notebook figures should be exported as report-ready images for the final submission.

The current Room 354 occupancy notebook covers sensor data from 2026-02-09 00:00 through 2026-03-10 13:35 after resampling to 5-minute intervals.

Within the notebook-derived Room 354 dataset, the exploratory feature levels were:

- VOC mean 85.74 with minimum 20.00, maximum 1044.56, and P90 150.15,
- CO2 mean 466.85 ppm with minimum 396.86, maximum 993.20, and P90 565.86,
- humidity mean 40.04 percent with minimum 15.60, maximum 59.50, and P90 54.30,
- temperature mean 70.58 F with minimum 64.42, maximum 79.41, and P90 72.73.

The notebook does not currently impute missing humidity or temperature values. Instead, it preserves missing values from the original sensor streams and uses whatever signals are available at each time step. This avoids inventing occupancy evidence where no measurement was recorded, but it also means that some periods have stronger support than others.

## 4. Methods and Tools

Methods and tools already reflected in the repository include:

- CSV-based HVAC and energy data collection,
- a database loading workflow for structured storage,
- notebook-based exploratory analysis,
- plotting and visualization of time-series sensor data,
- correlation analysis for IAQ feature relationships,
- rule-based occupancy estimation anchored to a CO2 mass-balance approximation.

At this point, the completed work is still exploratory rather than final-model driven.

Cleaning steps currently implemented in the Room 354 notebook:

- parse both Room 354 sensor timestamps into a shared datetime format,
- coerce VOC, CO2, humidity, and temperature fields to numeric values,
- resample both sensors to 5-minute intervals,
- merge the two Room 354 sensor streams on the shared time index,
- average overlapping CO2 and temperature readings across the two sensors to create unified room-level signals,
- retain missing values where one sensor does not report so the notebook can still use partial information.

Feature engineering currently implemented in the Room 354 notebook:

- unified `co2` feature created from the two Room 354 CO2 streams,
- unified `temperature` feature created from the available temperature streams,
- robust min-max normalization of CO2, VOC, humidity, and temperature using the 5th and 95th percentiles,
- a CO2-anchor estimate based on room volume, outdoor CO2, ACH reference, and per-person generation,
- a weighted multi-feature occupancy index with CO2 as the strongest signal, VOC as the next strongest signal, and humidity/temperature as supporting features,
- a blended per-timestamp `people_estimated` signal using 70 percent CO2 anchor and 30 percent multi-feature scaling,
- 3-point rolling smoothing of the final occupancy estimate for presentation.

Current occupancy estimation assumptions used in exploratory analysis:

- room dimensions: 50 ft x 30 ft x 15 ft,
- room volume: approximately 637.1 m^3,
- outdoor CO2 baseline: 420 ppm,
- per-person CO2 generation: 0.018 m^3/h/person (light activity),
- reference ACH scenario: 4 ACH for a central estimate.

Current exploratory modeling approach:

- use the Room 354 merged notebook dataset as the working analysis table,
- estimate occupancy continuously rather than assigning final labeled classes,
- treat the current output as a heuristic occupancy estimate rather than a validated final model because vibration and airflow data have not yet been integrated.

The current notebook uses Python with `pandas` for time parsing, numeric conversion, resampling, merging, and summary statistics; `matplotlib` and `seaborn` for plotting; and `numpy` for general numerical support. These tools were chosen because the problem is primarily time-series wrangling and exploratory visualization at this stage rather than large-scale model training.

Methods considered in the current notebook phase:

- direct visual comparison of VOC/CO2, humidity, and temperature,
- correlation analysis across the merged Room 354 features,
- a CO2-only occupancy anchor based on room volume and assumed ventilation parameters,
- a blended multi-feature estimate that scales and combines CO2, VOC, humidity, and temperature.

Methods not yet available in the current notebook:

- supervised classification or regression using labeled occupancy targets,
- validation against vibration data,
- validation against discharge airflow or room schedule information.

TODO: Add any additional methods the team tried outside the Room 354 notebook if they will remain part of the final project narrative.

## 5. Results

The current Room 354 notebook merges sensor data from 2026-02-09 00:00 through 2026-03-10 13:35 at 5-minute intervals and compares VOC/CO2, humidity, and temperature directly.

The 3-panel comparison indicates the following occupancy-related behavior:

- VOC vs CO2 Pearson correlation: 0.729 (strong positive),
- CO2 vs temperature Pearson correlation: 0.488 (moderate positive),
- VOC vs temperature Pearson correlation: 0.402 (moderate positive),
- humidity showed weaker direct relationships with VOC/CO2 in the current period.

Evaluation framing for this phase:

- no labeled occupancy ground truth is available yet, so accuracy-style metrics are not appropriate at this stage,
- the current baseline is the CO2-only anchor estimate,
- the current primary model is the blended estimate that combines CO2, VOC, humidity, and temperature,
- current evaluation is therefore descriptive: feature correlation, plot interpretation, and comparison of the baseline occupancy curve against the blended occupancy curve.

The baseline CO2-only anchor produced:

- mean estimate 6.90 people,
- P90 estimate 20.65 people,
- P95 estimate 30.31 people,
- P99 estimate 49.74 people,
- maximum estimate 81.16 people.

The blended occupancy estimate produced:

- mean estimated occupancy 7.25 people,
- P90 estimated occupancy 19.76 people,
- P95 estimated occupancy 28.43 people,
- P99 estimated occupancy 42.14 people,
- maximum estimated occupancy 63.72 people.

Interpretation of the notebook plots:

- CO2 and VOC show the clearest shared peaks and appear to be the strongest occupancy indicators in the current data,
- humidity changes more gradually over longer periods and appears more useful as a supporting feature than as a primary occupancy trigger,
- temperature remains comparatively stable for most of the study period and is best treated as contextual support rather than a standalone occupancy signal,
- compared with the CO2-only baseline, the blended estimate preserves major peaks while reducing some of the largest CO2-only spikes,
- the largest occupancy spikes should be treated cautiously until vibration and airflow data are added for confirmation.

Results currently supported directly by the notebook visualizations:

- the Room 354 notebook now includes a simple 3-panel comparison of VOC/CO2, humidity, and temperature,
- the notebook also includes a separate estimated occupancy graph so the inferred people-count trend can be read independently from the raw sensor trends.

![Figure 1. Room 354 3-panel time-series comparison of VOC/CO2, humidity, and temperature generated from the occupancy notebook.](figures/room354_feature_comparison.png)

![Figure 2. Room 354 estimated occupancy over time generated from the blended notebook model.](figures/room354_estimated_occupancy.png)

Within the scope of the current Room 354 notebook, the strongest evidence for occupancy detection is the repeated joint rise of VOC and CO2 and the fact that the blended occupancy signal remains numerically plausible while being less extreme than the CO2-only baseline. However, this should still be treated as a heuristic result rather than validated headcount estimation.

TODO: Add validated performance metrics once vibration data, airflow data, or labeled occupancy windows are available.

## 6. Conclusions and Future Work

The Room 354 occupancy notebook shows that VOC and CO2 are the strongest current occupancy indicators in the available indoor air quality data, while humidity and temperature act as supporting context. The current blended model produces an interpretable occupancy estimate over time and gives a more stable result than the CO2-only baseline, but it is not yet a validated occupancy-count model because no ground-truth labels have been incorporated.

The next steps are to strengthen the estimate with additional evidence and validation:

- integrate TDMS vibration data and align it to the same 5-minute time index used for the Room 354 notebook,
- add airflow or discharge measurements to better separate occupancy effects from HVAC dilution effects,
- create evaluation windows or occupancy labels so the CO2-only baseline and blended model can be compared more formally,
- convert the current continuous heuristic estimate into low, medium, and high occupancy classes if that representation is more useful for the final deliverable.

TODO: Refine this section once validation data has been added so the final conclusion can summarize actual measured performance rather than exploratory plausibility.

## 7. Appendix

Repository and reproducibility references:

- GitHub repository: https://github.com/DaltonSloan/CSC_4260_project
- Room 354 occupancy notebook: https://github.com/DaltonSloan/CSC_4260_project/blob/visuals/room354_occupancy_visualization.ipynb
- Report source: https://github.com/DaltonSloan/CSC_4260_project/blob/visuals/reports/updated_project_report.md
- Room 354 source data file 1: https://github.com/DaltonSloan/CSC_4260_project/blob/visuals/data/354.csv
- Room 354 source data file 2: https://github.com/DaltonSloan/CSC_4260_project/blob/visuals/data/354_2.csv
- Generated merged dataset used by the notebook: https://github.com/DaltonSloan/CSC_4260_project/blob/visuals/reports/room354_multisensor_merged.csv

Current reproducibility notes:

- the notebook is contained in the same repository as the data and report source,
- the current figures and statistics can be regenerated from the Room 354 notebook using the tracked CSV files,
- the repository also contains the merged Room 354 output file created during exploratory analysis.

TODO: Verify that the `visuals` branch links above remain publicly accessible at final submission time, or replace them with the final merged branch links.

TODO: Add any required team contribution breakdown or communication notes here if the instructor expects them inside the report rather than only in the repository documentation.
