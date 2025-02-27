# HospitalPredictionAnalysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains a machine learning project analyzing hospital performance and predicting patient `Test Results` (Normal, Abnormal, Inconclusive) using the `healthcare_dataset.csv` from Kaggle. The project demonstrates an end-to-end ML workflow—data preprocessing, exploratory analysis, feature engineering, model development, and deployment planning—achieving a peak accuracy of **64.08%** with XGBoost.

### Key Features
- **Data Preprocessing**: Standardized text, removed negative billing, calculated `Length of Stay`.
- **Exploratory Analysis**: Analyzed medical conditions, billing, and stay distributions.
- **Feature Engineering**: Added `Avg Billing by Condition`, `Age x Medical Condition`, `Billing Amount x Length of Stay`.
- **Models**:
  - XGBoost: **64.08%** accuracy (SMOTE, tuned: `n_estimators=200`, `max_depth=7`, `learning_rate=0.2`).
  - LightGBM: **62.24%** accuracy.
  - CatBoost: Tested with ongoing evaluation.
- **Class Imbalance**: Addressed with SMOTE, improving Abnormal/Inconclusive predictions.

## Dataset
The dataset (`healthcare_dataset.csv`) includes 55,500 patient records with 15 features:
- `Name`, `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Date of Admission`, `Doctor`, `Hospital`, `Insurance Provider`, `Billing Amount`, `Room Number`, `Admission Type`, `Discharge Date`, `Medication`, `Test Results`.

## Results
- **Best Model**: XGBoost with SMOTE and interaction features:
  - Accuracy: **64.08%**
  - Normal: 0.86 precision, 0.91 recall, 0.88 F1
  - Abnormal: ~0.52 F1
  - Inconclusive: ~0.51 F1
- **Insights**: `Billing Amount`, `Length of Stay`, and their interactions are key predictors.

## Deployment Plan
Upon completion, the model can be deployed for real-time hospital use:
1. **Serialization**: Save with `joblib` (e.g., `joblib.dump(xgb_model, 'hospital_xgb_model.pkl')`).
2. **API**: Flask/FastAPI service:
   - Input: Patient data (e.g., Age, Billing Amount).
   - Processing: Standardize, compute interactions.
   - Output: JSON prediction (e.g., `{"prediction": "Normal", "probability": 0.88}`).
3. **Cloud**: Deploy on AWS Lambda or Heroku.
4. **Monitoring**: Log predictions, retrain via CI/CD (e.g., GitHub Actions).

## Installation
To run locally:
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/HospitalPredictionAnalysis.git
   cd HospitalPredictionAnalysis
