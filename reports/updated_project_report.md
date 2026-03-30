# Updated Project Report

Team Members: Samuel Hartmann, Fengjun Han, Garrett Green, Dalton Sloan

Domain Experts: Chandler Norman, Norman Walker, Elisabeth Humphrey, Dr. Steven Anton

## 1. Problem Statement

The original project plan focused on evaluating HVAC efficiency in the Ashraf Islam Engineering Building at Tennessee Tech.

After exploratory data analysis, the project objective was revised. The current objective is to determine occupancy using VOC, temperature, humidity, and floor vibrations.

This change was made after reviewing the exploratory work and deciding that occupancy detection is the more efficient focus for the data currently available to the team.

TODO: Add the final team-approved wording for the updated problem statement.

TODO: Add a short explanation of why this occupancy objective matters for the building or the project goals.

## 2. Data and Exploratory Analysis

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
- creating an initial notebook that plots discharge air temperature and zone temperature for 361 data.

TODO: Add any additional exploratory plots the team has already made outside the current repository.

TODO: Add notes from any domain expert conversations that have already happened.

TODO: Add screenshots or exported figures from the original notebooks if they are needed in the final submission.

## 3. Methods and Tools

Methods and tools already reflected in the repository include:

- CSV-based HVAC and energy data collection,
- a database loading workflow for structured storage,
- notebook-based exploratory analysis,
- plotting and visualization of time-series sensor data.

At this point, the completed work is still exploratory rather than final-model driven.

TODO: Add the exact cleaning steps the team has already used.

TODO: Add the feature engineering steps the team has already completed.

TODO: Add the final modeling approach once the team selects it.

TODO: Add how occupancy will be labeled or estimated for evaluation.

## 4. Preliminary Results

The work currently present in the repository supports the following completed progress:

- whole-building energy consumption has been plotted over time,
- a 24-hour moving average has been generated for whole-building energy,
- discharge air temperature and zone temperature have been plotted for 361 data,
- the current datasets include fields related to the updated occupancy objective, including VOC, temperature, humidity, and a vibration-related activity proxy through the noise-level field.

TODO: Add the team’s written interpretation of the whole-building energy trend plot.

TODO: Add the team’s written interpretation of the 361 temperature plot.

TODO: Add occupancy-specific exploratory results for VOC, humidity, and floor vibrations once the team finishes that analysis.

TODO: Add any quantitative summaries that the team has already reviewed and approved.

## 5. Professionalism and Reproducibility

The current repository already shows:

- organized project data files,
- saved exploratory notebooks,
- a repeatable data-loading workflow,
- version-controlled project materials.

TODO: Add the team’s contribution breakdown.

TODO: Add any required note about meetings, expert feedback, or communication process.

TODO: Add the final list of submission artifacts once the team confirms them.

## 6. Next Steps

TODO: Add the next steps the team has agreed on for occupancy analysis.

TODO: Add the remaining work needed for the notebook.

TODO: Add the remaining work needed for the final report or poster.
