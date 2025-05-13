# Distributed ECG Analysis and Heart Attack Prediction with PySpark

This project applies distributed data analysis and machine learning techniques to a medical dataset of ECG recordings. The goal is to extract meaningful features and predict the presence of ischemic heart disease (heart attacks) using scalable tools.

---

## Overview

- Dataset: 2,500+ ECG records in `.hea` and `.mat` formats
- Task: Predict whether a patient has had a heart attack (binary classification)

---

## Project Structure

### `1_dataframe_creation.ipynb`
- Parsed and merged `.hea` and `.mat` files into a unified PySpark DataFrame
- Managed large arrays using custom dictionaries and partition tuning
- Cleaned constant fields and filtered by measurement consistency

### `2_data_preparation_understanding.ipynb`
- Converted derivations to numerical arrays
- Engineered features (mean, std) from ECG leads grouped by anatomical region
- Added HR-based features using **NeuroKit2** (e.g., BPM mean, R-peak detection)
- Visualizations: ECG waves, age distributions, average BPM, correlation matrix

### `3_classification.ipynb`
- Target variable: Heart attack occurrence (from diagnostic codes)
- Models:  
  - **Random Forest** (with Grid Search, 5-fold CV, and feature importance)
  - **Multilayer Perceptron (MLP)** with tuned layer architecture
- Addressed class imbalance with oversampling
- Best results (RF on balanced data):
  - Accuracy: 0.93  
  - AUC: 0.97  
  - Precision: 0.92  
  - Recall: 0.98

---

## Authors

- Camilla Chiruzzi
- Guido Trentacapilli
- Mathilde Clot  
- Niccolò Seghieri  
- Pierfrancesco Benincasa  


Project for **Distributed Data Analysis and Mining**  
University of Pisa – M.Sc. in Data Science and Business Informatics  

---
