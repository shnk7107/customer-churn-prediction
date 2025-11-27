# 1_train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os

DATA = "cleaned_telco.csv"
os.makedirs("models", exist_ok=True)

def load_data():
    df = pd.read_csv(DATA)
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return X, y

def evaluate_model(m, X_test, y_test):
    y_pred = m.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=4),
        "confusion": confusion_matrix(y_test, y_pred).tolist()
    }

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        print(f"--- {name} ---")
        print("Accuracy:", res['accuracy'])
        print("Precision:", res['precision'])
        print("Recall:", res['recall'])
        print("F1:", res['f1'])
        print(res['report'])
        # Save each model
        joblib.dump(model, f"models/{name}.pkl")

    # Choose best by f1 (you can change metric to accuracy)
    best_name = max(results.items(), key=lambda kv: kv[1]['f1'])[0]
    print("Best model by F1:", best_name)
    best_model = joblib.load(f"models/{best_name}.pkl")
    joblib.dump({"model": best_model, "name": best_name}, "models/best_model.pkl")
    print("Saved best model as models/best_model.pkl")

    # Save results summary
    pd.DataFrame({k: {m: v for m, v in results[k].items() if m in ['accuracy','precision','recall','f1']} for k in results}).to_csv("models/results_summary.csv")
    print("Saved results summary to models/results_summary.csv")

if __name__ == "__main__":
    main()
