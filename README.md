📘 Customer Churn Prediction (Telco Dataset)
Author: Shashank Tadikamalla
Course: Machine Learning Laboratory
Submitted to: Dr. Anjula Mehto
Date: November 2025

📌 Project Overview
This project predicts whether a customer will churn (leave the service) using the Telco Customer Churn Dataset from Kaggle.
The project includes:
Data Cleaning
Exploratory Data Analysis (EDA)
Model training (Logistic Regression, Random Forest, XGBoost)
Model evaluation using accuracy, precision, recall & F1-score
A Flask web demo for quick churn prediction
The goal is to build a simple, interpretable pipeline that can help businesses identify at-risk customers.

📂 Repository Structure
customer-churn-prediction/
│
├── 0_data_prep_and_eda.py        # Preprocessing + EDA (generates outputs/)
├── 1_train_models.py             # Trains LR, RF, XGBoost; saves results_summary.csv
├── 2_flask_demo.py               # Flask demo for predictions
│
├── cleaned_telco.csv             # Cleaned dataset
│
├── outputs/                      # EDA plots
│   ├── churn_counts.png
│   ├── dist_tenure.png
│   ├── dist_MonthlyCharges.png
│   └── dist_TotalCharges.png
│
├── models/
│   └── results_summary.csv       # Performance metrics of all models
│
├── train_log.txt                 # Console output of model training
├── requirements.txt              # Dependencies
├── Report.docx                   # Final report
└── README.md                     # This file
Note:
venv/ and large .pkl model files are intentionally removed to keep the repository lightweight.
Models can be recreated by running 1_train_models.py.

🛠️ Installation & Setup
1️⃣ Create and activate virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

2️⃣ Install required libraries
pip install -r requirements.txt

🚀 Running the Project
1️⃣ Data Preparation + EDA
python 0_data_prep_and_eda.py
Generates:
cleaned_telco.csv
EDA plots in outputs/

2️⃣ Model Training
python 1_train_models.py
This creates:
A summary of all model metrics: models/results_summary.csv
Training logs in train_log.txt

3️⃣ Flask Churn Prediction Demo
python 2_flask_demo.py
Then open:
👉 http://127.0.0.1:5000
This displays a simple UI that predicts churn using a sample row from the dataset.

📊 Model Performance (Summary)
| Model                   | Accuracy   | Precision  | Recall     | F1 Score   |
| ----------------------- | ---------- | ---------- | ---------- | ---------- |
| **Logistic Regression** | **0.7981** | **0.6355** | **0.5641** | **0.5977** |
| Random Forest           | 0.7882     | 0.6301     | 0.4919     | 0.5525     |
| XGBoost                 | 0.7782     | 0.5890     | 0.5481     | 0.5678     |
➡ Best Model (by F1-score): Logistic Regression

📈 Exploratory Data Analysis (EDA)
Included in the outputs/ folder:
churn_counts.png — Distribution of churned vs non-churned customers
dist_tenure.png — Tenure distribution
dist_MonthlyCharges.png — Monthly charges distribution
dist_TotalCharges.png — Total charges distribution
These plots help understand customer behavior trends.

🧾 Report
The report (Report.docx) includes:
Introduction
Problem Statement
Dataset Description
Data Preprocessing
EDA with screenshots
Model Results with metrics
Flask demo screenshots
Conclusion & References

🚀 Future Work
Apply SMOTE to handle class imbalance
Hyperparameter tuning for better performance
Add feature importance (SHAP/LIME)
Deploy Flask model to cloud (Render/AWS/Heroku)

🎉 End of Project
Thank you for reviewing this repository!
