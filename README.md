# HospitalPredictionAnalysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains a machine learning project analyzing hospital performance and predicting patient `Test Results` (Normal, Abnormal, Inconclusive) using the `healthcare_dataset.csv` dataset from Kaggle. The project showcases an end-to-end ML workflow, including data preprocessing, feature engineering, model training, and a deployment plan, achieving a peak accuracy of **0.643** with XGBoost.

### Key Features
- **Data Preprocessing**: Standardized text, removed negative billing, calculated `Length of Stay`.
- **Feature Engineering**: Added interaction terms (e.g., `Age x Medical Condition`, `Billing Amount x Length of Stay`).
- **Models**: Tested Random Forest (0.436), LightGBM+SMOTE (0.636), CatBoost (0.627), and XGBoost (0.643).
- **Evaluation**: Detailed classification reports with precision, recall, and F1-scores.
- **Work in Progress**: Targeting >0.70 accuracy with plans for further tuning and ensembling.

## Dataset
The project uses `healthcare_dataset.csv` from Kaggle, containing 55,500 patient records with 15 features:
- `Name`, `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Date of Admission`, `Doctor`, `Hospital`, `Insurance Provider`, `Billing Amount`, `Room Number`, `Admission Type`, `Discharge Date`, `Medication`, `Test Results`.

## Results
- **Best Model**: XGBoost with SMOTE and interaction features:
  - Accuracy: 0.643
  - Normal: 0.88 F1-score
  - Abnormal: 0.52 F1-score
  - Inconclusive: 0.51 F1-score
- **Insights**: `Billing Amount` and `Length of Stay` are key predictors, suggesting operational focus areas for hospitals.

## Deployment Plan
Once completed, the model can be deployed as follows:
1. **Serialization**: Save using `joblib` (e.g., `joblib.dump(xgb_model, 'xgb_model.pkl')`).
2. **API**: Flask/FastAPI service accepting patient data and returning predictions (e.g., `{"prediction": "Normal", "probability": 0.88}`).
3. **Cloud**: Host on AWS Lambda or Heroku.
4. **Monitoring**: Log predictions and retrain with a CI/CD pipeline.

## Installation
To run the notebook locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/mahlodi-makobe/HospitalPredictionAnalysis.git
   cd HospitalPredictionAnalysis
