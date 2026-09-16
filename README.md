````markdown
# Customer Churn Prediction

A machine learning project that predicts customer churn using the Telco Customer Churn dataset. The project covers the complete workflow from data preprocessing and exploratory data analysis to model training, evaluation, model selection, and deployment using Flask.

## Overview

Customer churn prediction helps organizations identify customers who are likely to discontinue their services. This project builds and compares multiple classification models to predict whether a customer will churn.

### Project Workflow

```text
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
One-Hot Encoding
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Model Serialization
     ↓
Flask Prediction Demo
````

## Dataset

The project uses the **Telco Customer Churn** dataset containing customer information related to:

* Demographics
* Customer tenure
* Services subscribed
* Contract details
* Billing information
* Monthly and total charges
* Churn status

The target variable is:

```text
Churn
0 → Customer retained
1 → Customer churned
```

## Machine Learning Models

The following classification algorithms were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The **F1 Score** was used as the primary criterion for selecting the final model.

## Results

| Model               | Accuracy | Precision | Recall |   F1 Score |
| ------------------- | -------: | --------: | -----: | ---------: |
| Logistic Regression |   79.82% |    63.55% | 56.42% | **59.77%** |
| Random Forest       |   78.82% |    63.01% | 49.20% |     55.26% |
| XGBoost             |   77.83% |    58.91% | 54.81% |     56.79% |

**Selected Model:** Logistic Regression

## Data Preprocessing

The preprocessing pipeline includes:

* Converting `TotalCharges` to numeric format
* Handling missing values
* Removing the `customerID` identifier
* Encoding the `Churn` target variable
* Feature engineering
* One-hot encoding categorical features
* Preparing the dataset for machine learning

## Exploratory Data Analysis

The project performs exploratory analysis to understand customer churn patterns and feature distributions.

Generated visualizations include:

* Churn distribution
* Monthly Charges distribution
* Total Charges distribution
* Customer Tenure distribution

The generated outputs are available in the `outputs/` directory.

## Flask Prediction Demo

A Flask web application is included to demonstrate real-time model inference.

The application:

1. Loads the trained model.
2. Loads the processed customer data.
3. Allows selection of a customer record.
4. Generates a churn prediction.
5. Displays the predicted churn status.
6. Displays the prediction probability.

Run the Flask application:

```bash
python 2_flask_demo.py
```

Then open:

```text
http://localhost:5000
```

## Project Structure

```text
customer-churn-prediction/
│
├── models/
│
├── outputs/
│
├── 0_data_prep_and_eda.py
├── 1_train_models.py
├── 2_flask_demo.py
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── cleaned_telco.csv
├── ccp_ml_report.pdf
├── train_log.txt
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/shnk7107/customer-churn-prediction.git
cd customer-churn-prediction
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1 — Data Preparation & EDA

```bash
python 0_data_prep_and_eda.py
```

### Step 2 — Train Models

```bash
python 1_train_models.py
```

### Step 3 — Run Flask Application

```bash
python 2_flask_demo.py
```

Open:

```text
http://localhost:5000
```

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Flask**

## Skills Demonstrated

* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Categorical Encoding
* Machine Learning Classification
* Model Training & Comparison
* Model Evaluation
* Model Selection
* Model Serialization
* Flask Application Development
* End-to-End ML Pipeline Development

## Author

**Shashank Tadikamalla**

Machine Learning Laboratory Project
November 2025
