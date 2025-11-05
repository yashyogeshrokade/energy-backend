from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
with open("rf_energy_model_final.pkl", "rb") as f:
    model = pickle.load(f)

# Define expected input features in correct order
FEATURES = [
    "temperature", "humidity", "AC", "Labs", "AHU", "Servers",
    "edu", "admin", "res", "dining"
]

@app.route("/")
def home():
    return "✅ Energy Consumption Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Example input: {"temperature":29, "humidity":60, "AC":1, "Labs":0, "AHU":0, "Servers":1, "building_type":"educational"}

        temp = float(data.get("temperature", 0))
        hum = float(data.get("humidity", 0))

        # Binary load features
        AC = int(data.get("AC", 0))
        Labs = int(data.get("Labs", 0))
        AHU = int(data.get("AHU", 0))
        Servers = int(data.get("Servers", 0))

        # One-hot encode building type
        btype = data.get("building_type", "").lower()
        edu = 1 if btype == "educational" else 0
        admin = 1 if btype == "administration" else 0
        res = 1 if btype == "residential" else 0
        dining = 1 if btype == "dining" else 0

        # Make dataframe for model
        X = pd.DataFrame([[temp, hum, AC, Labs, AHU, Servers, edu, admin, res, dining]], columns=FEATURES)

        # Predict
        y_pred = model.predict(X)[0]

        return jsonify({
            "success": True,
            "predicted_energy": round(float(y_pred), 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
