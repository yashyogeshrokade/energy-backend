from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# Restrict CORS to your site for production; for now allow all
CORS(app)  # or CORS(app, origins=["https://yashyogeshrokade.github.io"])

# --- load model ---
MODEL_PATH = "rf_energy_model_final.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Load reference Excel (placed in same repo) ---
EXCEL_PATH = "train_ready_combined_IIIT_IITM_temp_adjusted.xlsx"
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Reference Excel not found at {EXCEL_PATH}. Upload it to the backend repo.")

# read Excel once at start
building_data = pd.read_excel(EXCEL_PATH)

# normalize column names for easier lookup
colmap = {c.lower(): c for c in building_data.columns}

# helper to find possible column names
def find_col(possible_names):
    for name in possible_names:
        if name.lower() in colmap:
            return colmap[name.lower()]
    return None

# probable columns (adjust if your excel uses other names)
COL_BUILDING = find_col(["Building_Name", "building_name", "Building", "building", "Name"])
COL_CLUSTER = find_col(["Cluster_ID", "cluster_id", "Cluster", "cluster"])
COL_TOTAL_AREA = find_col(["Total_Area_ft2", "total_area_ft2", "TotalArea_ft2"])
COL_HVAC_AREA = find_col(["HVAC_Area_ft2", "hvac_area_ft2"])
COL_OCC = find_col(["Occupancy_Encoded", "occupancy_encoded", "Occupancy", "occupancy"])
# special feature columns (use the training names you used)
COL_SPECIAL_LAB = find_col(["Special_Lab", "special_lab"])
COL_SPECIAL_AC = find_col(["Special_AC", "special_ac", "Special_AC"])
COL_SPECIAL_SERVERS = find_col(["Special_Servers", "special_servers"])
COL_SPECIAL_AHU = find_col(["Special_AHU", "special_ahu"])
COL_DINING = find_col(["dining"])
COL_EDU = find_col(["educational", "Educational"])
COL_ADMIN = find_col(["adminstration","administration","admin"])  # sample used adminstration typo
COL_RES = find_col(["residential", "Residential"])

# Decide final feature order that the model expects (based on your training sample)
# Use names exactly as your model expects. The sample_data you showed used these column names:
FEATURES = [
    "Year",
    "Total_Area_ft2",
    "HVAC_Area_ft2",
    "Avg_Temp_C",
    "Avg_Humidity_%",
    "Occupancy_Encoded",
    "Special_Lab",
    "Special_AC",
    "Special_Servers",
    "dining",
    "educational",
    "adminstration",
    "residential",
    "Special_AHU"
]

@app.route("/")
def home():
    return "✅ Energy Consumption Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        # read frontend inputs
        building_in = str(payload.get("building", "")).strip()
        temperature = payload.get("temperature", None)
        humidity = payload.get("humidity", None)
        month = str(payload.get("month", "")).strip().lower()

        if building_in == "" or temperature is None or humidity is None or month == "":
            return jsonify({"success": False, "error": "Missing building, temperature, humidity or month in request"}), 400

        # find building row (case-insensitive)
        if COL_BUILDING is None:
            return jsonify({"success": False, "error": "Could not find building name column in reference Excel"}), 500

        # prepare lowercase comparison
        building_lower = building_in.lower()
        building_data['_bnorm'] = building_data[COL_BUILDING].astype(str).str.strip().str.lower()
        matched = building_data[building_data['_bnorm'] == building_lower]

        if matched.empty:
            # try fuzzy match: contains
            matched = building_data[building_data['_bnorm'].str.contains(building_lower, na=False)]
            if matched.empty:
                return jsonify({"success": False, "error": f"Building '{building_in}' not found in Excel (searched column {COL_BUILDING})"}), 400

        # take the first match row for base features
        base_row = matched.iloc[0]

        # Determine occupancy according to month rule
        # months with forced occupancy 1:
        seasonal_months = {"january", "may", "june", "december"}
        if month.lower() in seasonal_months:
            occupancy_value = 1
            occupancy_source = "seasonal_forced"
        else:
            # use cluster's 3rd row occupancy (index 2) if cluster exists
            if COL_CLUSTER is not None:
                cluster_id = base_row[COL_CLUSTER]
                cluster_group = building_data[building_data[COL_CLUSTER] == cluster_id]
                if len(cluster_group) >= 3:
                    occupancy_value = cluster_group.iloc[2][COL_OCC] if COL_OCC is not None else base_row[COL_OCC]
                    occupancy_source = "cluster_third_row"
                else:
                    # fallback to building's own occupancy
                    occupancy_value = base_row[COL_OCC] if COL_OCC is not None else 1
                    occupancy_source = "fallback_building_row"
            else:
                # no cluster column: use building's occupancy
                occupancy_value = base_row[COL_OCC] if COL_OCC is not None else 1
                occupancy_source = "no_cluster_column"

        # Build feature dict in the exact order and names the model expects.
        # If the Excel uses slightly different column names, we map them into expected keys.
        features_dict = {}

        # Year fixed to 2025 as you requested
        features_dict["Year"] = 2025

        # Total area & HVAC area
        features_dict["Total_Area_ft2"] = float(base_row[COL_TOTAL_AREA]) if COL_TOTAL_AREA is not None else 0.0
        features_dict["HVAC_Area_ft2"] = float(base_row[COL_HVAC_AREA]) if COL_HVAC_AREA is not None else 0.0

        # temperature/humidity go to Avg_Temp_C / Avg_Humidity_%
        features_dict["Avg_Temp_C"] = float(temperature)
        features_dict["Avg_Humidity_%"] = float(humidity)

        # occupancy
        features_dict["Occupancy_Encoded"] = float(occupancy_value)

        # special features
        features_dict["Special_Lab"] = int(base_row[COL_SPECIAL_LAB]) if COL_SPECIAL_LAB is not None else 0
        features_dict["Special_AC"] = int(base_row[COL_SPECIAL_AC]) if COL_SPECIAL_AC is not None else 0
        features_dict["Special_Servers"] = int(base_row[COL_SPECIAL_SERVERS]) if COL_SPECIAL_SERVERS is not None else 0

        # building type one-hot flags (some of these may already be present)
        features_dict["dining"] = int(base_row[COL_DINING]) if COL_DINING is not None else 0
        features_dict["educational"] = int(base_row[COL_EDU]) if COL_EDU is not None else 0
        # keep the exact spelling used in training ("adminstration" in sample) - try both
        if COL_ADMIN is not None:
            features_dict["adminstration"] = int(base_row[COL_ADMIN])
        else:
            # if not present, try to infer from building_type or set 0
            features_dict["adminstration"] = 0
        features_dict["residential"] = int(base_row[COL_RES]) if COL_RES is not None else 0

        features_dict["Special_AHU"] = int(base_row[COL_SPECIAL_AHU]) if COL_SPECIAL_AHU is not None else 0

        # Build DataFrame in the same order as FEATURES
        X_row = [features_dict.get(k, 0) for k in FEATURES]
        X = pd.DataFrame([X_row], columns=FEATURES)

        # Predict with the model
        y_pred = model.predict(X)[0]

        # For debugging we return occupancy used and which features we passed (optionally)
        return jsonify({
            "success": True,
            "predicted_energy": round(float(y_pred), 2),
            "occupancy_used": float(features_dict["Occupancy_Encoded"]),
            "occupancy_source": occupancy_source,
            "features_used": FEATURES
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
