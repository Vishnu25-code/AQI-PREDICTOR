# AQI Real-Time Data Scaling Project
## Overview

This notebook processes real-time air quality pollutant data and scales it to match training dataset statistics. It prepares live AQI features for reliable machine learning prediction and analysis.

### Features

+ Loads real-time pollutant data into a DataFrame

+ Uses predefined training means

+ Scales each feature column to match training distribution

+ Prevents divide-by-zero and missing-column errors

+ Exports processed dataset to CSV

### Pollutants Used

+ PM2.5, PM10, NO₂, SO₂, CO, O₃

### Core Logic

+ Each feature is rescaled using:

+ scaled_value = value × (training_mean / live_mean)


This aligns real-time inputs with model training conditions.

## How to Run

+ Open the notebook

+ Restart kernel

+ Run all cells in order

+ Ensure df_real is created before scaling step

**Check generated output file**

### Common Issue

NameError: df_real not defined
Occurs when the data-loading cell was not executed.
Fix: Restart kernel and run all cells.
