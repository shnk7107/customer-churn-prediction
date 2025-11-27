# 0_data_prep_and_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

IN = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = "cleaned_telco.csv"
os.makedirs("outputs", exist_ok=True)

def load_and_clean(path=IN):
    df = pd.read_csv(path)
    # fix total charges: convert blanks to NaN then numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # drop customerID
    df = df.drop(columns=['customerID'])
    # drop rows with missing total charges (if few)
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    # Convert churn to binary
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    # Convert SeniorCitizen 0/1 to string
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0:'No', 1:'Yes'})
    return df

def feature_engineering(df):
    # Convert multiple categorical columns to category type
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Replace spaces in column names
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    # One-hot encoding for categorical variables with reasonable cardinality
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def eda(df):
    # Simple churn rate
    churn_rate = df['Churn'].mean()
    print(f"Churn rate: {churn_rate:.3f}")
    # Plot churn counts
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn counts (0 = No, 1 = Yes)')
    plt.savefig("outputs/churn_counts.png", bbox_inches='tight')
    plt.close()

    # Top numeric features distribution
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    subset = num_cols[:6]
    for c in subset:
        plt.figure(figsize=(6,3))
        sns.histplot(df[c], kde=False)
        plt.title(f'Distribution: {c}')
        plt.savefig(f"outputs/dist_{c}.png", bbox_inches='tight')
        plt.close()

def main():
    df = load_and_clean()
    df.to_csv("outputs/raw_cleaned.csv", index=False)
    eda(df)
    df_fe = feature_engineering(df)
    df_fe.to_csv(OUT, index=False)
    print("Saved cleaned data to", OUT)
    print("EDA plots saved in outputs/")

if __name__ == "__main__":
    main()
