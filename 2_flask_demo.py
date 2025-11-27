# 2_flask_demo.py
from flask import Flask, render_template_string, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "cleaned_telco.csv"

# load model (assumes you already ran training and models/best_model.pkl exists)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run 1_train_models.py first.")
data = joblib.load(MODEL_PATH)
model = data["model"]

# load cleaned data (this file was produced by 0_data_prep_and_eda.py)
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c != "Churn"]

HTML = """
<!doctype html>
<title>Customer Churn Prediction (Demo)</title>
<h2>Customer Churn Prediction</h2>
<p>Select a sample row from the cleaned dataset to predict (safe & exact).</p>

<form method="post">
  <label for="row">Pick row index (0 .. {{max_idx}}):</label>
  <input type="number" id="row" name="row" min="0" max="{{max_idx}}" value="0" required>
  <input type="submit" value="Predict">
</form>

{% if sample is not none %}
  <h3>Sample row (first 8 columns shown):</h3>
  <pre>{{ sample }}</pre>
{% endif %}

{% if pred is not none %}
  <h3>Predicted: {{ pred }} (probability: {{ prob }})</h3>
{% endif %}

<hr>
<p>Tips: If you want to test a custom row manually, open <code>cleaned_telco.csv</code> in VS Code,
copy a row, and paste values back into the CSV or use its row index above.</p>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    prob = None
    sample_text = None
    max_idx = len(df) - 1
    if request.method == "POST":
        try:
            idx = int(request.form["row"])
            if idx < 0 or idx > max_idx:
                raise ValueError("row index out of range")
            row = df.iloc[idx]
            sample_text = row.head(8).to_string()  # show first 8 cols for readability
            X = row.drop(labels=["Churn"], errors="ignore").values.reshape(1, -1)
            p = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            pred = "Yes (1)" if int(p) == 1 else "No (0)"
            prob = f"{probs[int(p)]:.4f}"
        except Exception as e:
            pred = f"Error: {e}"
    return render_template_string(HTML, pred=pred, prob=prob, sample=sample_text, max_idx=max_idx)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
