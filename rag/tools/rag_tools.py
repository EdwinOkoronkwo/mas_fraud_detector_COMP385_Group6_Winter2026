import json
import re
import sqlite3
from langchain.tools import tool
from autogen_ext.tools.langchain import LangChainToolAdapter

import joblib
import numpy as np
import os
from config.settings import settings

import csv
from datetime import datetime



LOG_FILE = "mas_fraud_detector/logs/inference_audit.csv"


def log_inference_event(step: str, data: dict):
    """Saves every calculation to a CSV for audit and comparison."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "step", "details"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "details": str(data)
        })

# 1. Define the logic as a tool
@tool
def inspect_database_schema(db_path: str = r"C:\Youtube\autogen\mas_fraud_detector\data\database.sqlite") -> str:
    """
    Automatically discovers the tables and columns in the database.
    Use this first if you are unsure of the schema or if a query fails.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    schema_info = "Database Schema:\n"
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [f"{c[1]} ({c[2]})" for c in cursor.fetchall()]
        schema_info += f"- Table '{table}' has columns: {', '.join(cols)}\n"

    conn.close()
    return schema_info


# 2. Wrap it for AutoGen
autogen_inspect_tool = LangChainToolAdapter(inspect_database_schema)

from autogen_core.tools import FunctionTool

import joblib
import torch
import os
import numpy as np

MODEL_DIR = r"C:\Youtube\autogen\mas_fraud_detector\models"

import numpy as np
import joblib
import os




from typing import Union, List, Dict
import os
import joblib


def prepare_full_inference_vector(raw_row: Dict, model_dir: str = "models") -> Dict:
    """
    Combines scaling (9 features) and one-hot encoding (12 features)
    to create the final 21-feature vector expected by XGBoost.
    """
    # 1. Load the Scaler and the Model's feature list
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    # In your case, this should list the 21 features the model was trained on
    expected_21 = [
        'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'category_entertainment', 'category_food_dining', 'category_gas_transport',
        'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
        'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos',
        'category_personal_care', 'category_shopping_net', 'category_shopping_pos', 'category_travel'
    ]

    # 2. Scale the 9 Numeric Features
    numeric_cols = scaler.feature_names_in_
    numeric_input = [float(raw_row.get(f, 0)) for f in numeric_cols]
    scaled_values = scaler.transform([numeric_input])[0]
    final_payload = dict(zip(numeric_cols, scaled_values))

    # 3. Manual One-Hot Encoding for 'category'
    # Default all category columns to 0
    cat_prefix = "category_"
    current_cat = f"{cat_prefix}{raw_row.get('category', 'misc_pos')}"

    for col in expected_21:
        if col.startswith(cat_prefix):
            final_payload[col] = 1 if col == current_cat else 0

    return final_payload


def scale_transaction_data(features_input: Dict, model_dir: str = "models") -> Dict:
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    # The order MUST match what the scaler saw during fit_transform
    expected_columns = scaler.feature_names_in_

    # 1. Map raw input to the OHE columns (Category/Gender)
    # This prepares the 'raw' 21-feature row
    raw_row = {}
    active_cat = f"category_{features_input.get('category', 'misc_pos')}"
    active_gender = f"gender_{features_input.get('gender', 'M')}"

    for col in expected_columns:
        if col in features_input:
            raw_row[col] = float(features_input[col])
        elif col == active_cat or col == active_gender:
            raw_row[col] = 1.0
        else:
            raw_row[col] = 0.0

    # 2. Convert to the exact list/order required
    ordered_values = [raw_row[col] for col in expected_columns]

    # 3. Scale EVERYTHING (The scaler now handles the 0/1s too)
    # This ensures the dimensions match (21 in -> 21 out)
    scaled_values = scaler.transform([ordered_values])[0]

    # 4. Return as a dictionary for the Agents to read
    return dict(zip(expected_columns, scaled_values))




#
import os
import joblib
import numpy as np

# Set your model directory path here
MODEL_DIR = "models"




import json
import os
import joblib
import torch
import numpy as np

import os
import json


def load_champion_registry():
    # This gets the directory where rag_tools.py actually lives
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up levels if 'models' is in the project root and this script is in 'rag/tools/'
    # Assuming structure: mas_fraud_detector/models/champion_registry.json
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))

    registry_path = os.path.join(project_root, "models", "champion_registry.json")

    print(f"🔍 Attempting to load registry from: {registry_path}")

    if not os.path.exists(registry_path):
        # Fallback: Create a dummy registry if it's missing to prevent crashing
        return {"champions": {}, "ensemble_weights": {}}

    with open(registry_path, "r") as f:
        return json.load(f)


REGISTRY = load_champion_registry()
CHAMPIONS = REGISTRY.get("champions", {})
WEIGHTS = REGISTRY.get("ensemble_weights", {})


import torch
import torch.nn as nn


# 1. VAE Class Definition (Must match your training architecture)
class VAE(nn.Module):
    def __init__(self, input_dim=22, latent_dim=8):  # Use 22 to match the training truth
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            # The middle layers (20 and 12) are internal,
            # so they stay the same as long as they match your .pth file structure.
            nn.ReLU()
        )

        # This matches the [8, 12] shape from your earlier error log
        self.fc_mu = nn.Linear(12, latent_dim)
        self.fc_logvar = nn.Linear(12, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim)  # Final layer must output 22 to match input
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# 2. VAE Prediction Function

import torch
import torch.nn as nn
import os


# 1. Update the Class Definition
class Residual_LSTM_AE(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.enc1 = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.enc2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.dec1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dec2 = nn.LSTM(hidden_dim, in_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out1, _ = self.enc1(x)
        out1 = self.dropout(out1)
        _, (h2, _) = self.enc2(out1)
        latent = h2.permute(1, 0, 2)
        out3, _ = self.dec1(latent)
        out3 = self.dropout(out3)
        out4, _ = self.dec2(out3)
        # CRITICAL: Must match training architecture
        return out4 + x


def predict_baseline_shadow(scaled_data: dict) -> float:
    """
    Executes the manual XGB baseline for comparative reporting.
    This model represents the 'pre-AI' state of the project.
    """
    try:
        # 1. Load the manual baseline model
        # Path: models/baselines/manual_xgb_baseline.pkl
        model_path = os.path.join(settings.PROJECT_ROOT, "models", "baselines", "manual_xgb_baseline.pkl")
        model = joblib.load(model_path)

        # 2. Extract features in the EXACT order expected by the baseline
        # Even if the baseline only uses a subset, we align to its 'feature_names_in_'
        expected_cols = getattr(model, 'feature_names_in_', [
            "amt", "zip", "lat", "long", "city_pop", "merch_lat", "merch_long"
        ])

        # 3. Prepare the row
        input_row = np.array([[scaled_data.get(f, 0.0) for f in expected_cols]])

        # 4. Return probability
        # Using predict_proba for consistency with the other pillars
        return float(model.predict_proba(input_row)[0][1])
    except Exception as e:
        print(f"⚠️ Baseline Shadow failed: {e}")
        return 0.0




def predict_neural_pillar(scaled_data: dict) -> float:
    path = CHAMPIONS["neural"]

    # FIX: Get order from manifest (or hardcoded 21 list we verified)
    expected_order = [
        "amt", "zip", "lat", "long", "city_pop", "merch_lat", "merch_long",
        "category_food_dining", "category_gas_transport", "category_grocery_net",
        "category_grocery_pos", "category_health_fitness", "category_home",
        "category_kids_pets", "category_misc_net", "category_misc_pos",
        "category_personal_care", "category_shopping_net", "category_shopping_pos",
        "category_travel", "gender_M"
    ]

    # Explicitly map the values in order
    vals = [scaled_data[f] for f in expected_order]
    tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    # Load model (Dynamic selection based on path)
    if "vae" in path.lower():
        model = VAE(input_dim=len(vals))
        tensor_in = tensor
    else:
        model = Residual_LSTM_AE(in_dim=len(vals))
        tensor_in = tensor.unsqueeze(1)  # RNN Dim

    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    with torch.no_grad():
        if "vae" in path.lower():
            recon, mu, logvar = model(tensor) # VAE returns tuple
            mse = torch.mean((tensor - recon) ** 2).item()
        else:
            recon = model(tensor.unsqueeze(1))
            mse = torch.mean((tensor.unsqueeze(1) - recon) ** 2).item()

    # Probability Mapping
    threshold = 0.75 # Adjust based on your manifest's threshold_mse
    return float(mse)


def predict_clustering_pillar(scaled_data: dict) -> float:
    model = joblib.load(CHAMPIONS["clustering"])

    # FIX: Use explicit order
    expected_order = [
        "amt", "zip", "lat", "long", "city_pop", "merch_lat", "merch_long",
        "category_food_dining", "category_gas_transport", "category_grocery_net",
        "category_grocery_pos", "category_health_fitness", "category_home",
        "category_kids_pets", "category_misc_net", "category_misc_pos",
        "category_personal_care", "category_shopping_net", "category_shopping_pos",
        "category_travel", "gender_M"
    ]
    input_row = np.array([[scaled_data[f] for f in expected_order]])

    # If K-Means: Use transform to get distance to centroids
    if hasattr(model, 'transform'):
        distances = model.transform(input_row)
        return float(np.min(distances))  # Return the REAL distance

    return 0.0


def predict_neural_pillar_raw(scaled_data: dict) -> float:
    print("\n🧠 [LOG] Starting Neural Pillar Raw Prediction...")
    path = CHAMPIONS["neural"]

    # 1. THE BRIDGE: Define the exact 12 features the .pth file expects
    VAE_12_FEATURES = [
        "amt", "zip", "lat", "long", "city_pop", "unix_time",
        "merch_lat", "merch_long", "category_food_dining",
        "category_gas_transport", "category_grocery_net", "gender_M"
    ]

    try:
        # 2. Extract only these 12 from the 22-feature payload
        vals = [scaled_data[f] for f in VAE_12_FEATURES]
        tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
        print(f"📊 [LOG] Neural Input Tensor Shape: {tensor.shape} (Bridged 22 -> 12)")
    except KeyError as e:
        print(f"❌ [ERROR] Neural missing expected feature: {e}")
        return 0.0

    # 3. INITIALIZE WITH CHECKPOINT GEOMETRY
    # We use the inference class that matches the [8, 12] checkpoint
    model = VAE(input_dim=12, latent_dim=8)

    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"❌ [ERROR] State Dict Load Failed: {e}")
        return 0.0

    with torch.no_grad():
        recon, mu, logvar = model(tensor)
        mse = torch.mean((tensor - recon) ** 2).item()

    print(f"✅ [SUCCESS] Neural Reconstruction Error (MSE): {mse:.6f}")
    return float(mse)


def predict_supervised_pillar(scaled_data: dict) -> float:
    # ... Use the 9-feature Bridge ...
    # Features: ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'gender_M']
    print("\n⚔️ [LOG] Starting Supervised Pillar Prediction...")
    model = joblib.load(CHAMPIONS["supervised"])
    scaler = joblib.load(os.path.join(settings.PROJECT_ROOT, "models", "scaler.joblib"))

    # Trust the scaler's order implicitly
    expected_order = list(scaler.feature_names_in_)

    try:
        # Construct vector based on the EXACT order used during training
        # Inside your tool file:
        OLD_9_FEATURES = ["amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long", "gender_M"]
        input_row = np.array([[scaled_data[f] for f in OLD_9_FEATURES]])
        print(f"🧪 [LOG] Input Vector Shape: {input_row.shape} (Matches expected {len(expected_order)})")
    except KeyError as e:
        print(f"❌ [ERROR] Missing feature in Agent Payload: {e}")
        # This will trigger if the Agent filters the dictionary
        return 0.0

    probs = model.predict_proba(input_row)[0]
    return float(probs[1])




def predict_clustering_pillar_raw(scaled_data: dict) -> float:
    print("\n🔍 [LOG] Starting Clustering Pillar Raw Prediction...")

    # 1. Load Model & Scaler
    model = joblib.load(CHAMPIONS["clustering"])
    scaler_path = os.path.join("models", "scaler.joblib")
    scaler = joblib.load(scaler_path)

    # 2. Get the expected order from the scaler
    expected_order = scaler.feature_names_in_
    print(f"📊 [LOG] Model expects {len(expected_order)} features.")

    # 3. Construct input and check for missing keys
    try:
        # LOG: Check if any expected keys are missing from the input data
        missing = [f for f in expected_order if f not in scaled_data]
        if missing:
            print(f"❌ [ERROR] Missing features in input: {missing}")
            return 0.0

        input_row = np.array([[scaled_data[f] for f in expected_order]])

        # LOG: See the actual values being fed in (Truncated for readability)
        print(f"🧪 [LOG] Raw Input Vector (First 5): {input_row[0][:5]}")

    except Exception as e:
        print(f"❌ [ERROR] Construction failed: {e}")
        return 0.0

    # 4. Calculate Distance
    if hasattr(model, 'transform'):
        distances = model.transform(input_row)
        min_dist = float(np.min(distances))

        # LOG: The final result
        print(f"✅ [SUCCESS] Nearest Cluster Distance: {min_dist:.4f}")
        return min_dist

    print("⚠️ [WARNING] Model does not have 'transform' attribute.")
    return 0.0

import json


def run_baseline_prediction(feature_dict: dict) -> str:
    """
    Independent Baseline Prediction.
    Loads the config we saved to ensure column alignment.
    """
    BASE_DIR = "results/baseline_experiment"
    CONFIG_PATH = os.path.join(BASE_DIR, "inference_config.json")

    try:
        # 1. Load the Map (The JSON we just made)
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        # 2. Load the Model
        model_path = os.path.join(BASE_DIR, config["model_info"]["file"])
        model = joblib.load(model_path)

        # 3. Align Data (The "Magic" Step)
        # Convert the raw dictionary to a DataFrame using the EXACT order from training
        df = pd.DataFrame([feature_dict])
        ordered_df = df[config["inference_params"]["expected_features"]]

        # 4. Predict
        prob = float(model.predict_proba(ordered_df)[0][1])

        return f"BASELINE_XGB_SCORE: {prob:.4f}"

    except Exception as e:
        return f"BASELINE_ERROR: {str(e)}"

import os
from datetime import datetime


def build_audit_report(final_score: float, status: str, mode: str, s_score: float, n_score: float,
                       c_score: float) -> str:
    """
    Generates a professional Markdown-formatted audit report for the ensemble decision.
    """
    # 1. Visual indicator for the risk level
    severity_emoji = "🔴" if status == "CRITICAL" else "🟡" if status == "HIGH" else "🟢"

    # 2. Extract Model Names Safely
    # This handles paths like 'models/champion_xgb.joblib' or 'models/rnn_v2.pth'
    def _clean_name(key):
        path = CHAMPIONS.get(key, "Unknown")
        return os.path.basename(path).replace('champion_', '').split('.')[0].upper()

    # 3. Use standard Python datetime (os.popen('date') is slow and OS-dependent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
### {severity_emoji} FRAUD AUDIT REPORT: {status}
---
**FINAL AGGREGATED RISK SCORE:** `{final_score:.4f}`
**CALCULATION STRATEGY:** `{mode}`

| Pillar | Model Type | Contribution Score |
| :--- | :--- | :--- |
| **Supervised** | {_clean_name('supervised')} | {s_score:.4f} |
| **Neuro-Pattern** | {_clean_name('neural')} | {n_score:.4f} |
| **Clustering** | {_clean_name('clustering')} | {c_score:.4f} |

---
**DECISION SUMMARY:**
The system determined a **{status}** risk level. This calculation was performed using the **{mode}** logic, ensuring that high-confidence individual model "screams" (Consensus Overrides) or high-value transaction bypasses were accounted for.

*Timestamp: {timestamp}*
"""
    return report


def build_audit_report_with_baseline(final_score, status, mode, s_s, n_s, c_s, b_s):
    # Calculate Variance: How much better is the Ensemble than the Baseline?
    improvement = (final_score - b_s) * 100

    report = f"""
### 📊 ENSEMBLE VS. BASELINE COMPARISON
| Metric | Score | Performance Status |
| :--- | :--- | :--- |
| **Advanced AI Ensemble** | `{final_score:.4f}` | {status} |
| **Manual XGB Baseline** | `{b_s:.4f}` | REFERENCE |
| **Ensemble Variance** | `{improvement:+.2f}%` | {"🚀 LEAD" if improvement > 0 else "⚠️ UNDER"} |

---
**PILLAR BREAKDOWN:**
* **Supervised:** {s_s:.4f}
* **Neural:** {n_s:.4f}
* **Clustering:** {c_s:.4f}
"""
    return report



import pandas as pd
import numpy as np

import torch
import numpy as np
import joblib

# Configuration derived from your training logs
VAE_GEOMETRY = {"input": 22, "latent": 8}
RF_FEATURES = ["amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long", "gender_M"]


def execute_champion_ensemble(scaled_data: dict, raw_features: list):
    """
    Finalized Ensemble: Maps 22 features to VAE/DBSCAN and bridges 9 features to RF.
    """
    results = {"supervised": 0.0, "neural": 0.0, "clustering": 0.0, "total": 0.0}

    # 1. Prepare Vectors
    # Full 22-feature list in training order
    full_vector = np.array([[v for k, v in scaled_data.items()]]).astype(np.float32)

    # Bridged 9-feature list for the older RF model
    rf_vector = np.array([[scaled_data[f] for f in RF_FEATURES]])

    # --- PILLAR 1: SUPERVISED (RF) ---
    try:
        rf_model = joblib.load("models/champion_rf.joblib")
        results["supervised"] = float(rf_model.predict_proba(rf_vector)[0][1])
    except Exception as e:
        print(f"⚠️ RF Pillar Error: {e}")

    # --- PILLAR 2: NEURAL (VAE) ---
    try:
        # Initializing with the [8, 12] geometry found in your checkpoint
        model = VAE(input_dim=VAE_GEOMETRY["input"], latent_dim=VAE_GEOMETRY["latent"])
        model.load_state_dict(torch.load("models/champion_vae.pth", weights_only=True))
        model.eval()

        with torch.no_grad():
            tensor_in = torch.from_numpy(full_vector)
            recon, mu, logvar = model(tensor_in)
            # MSE as a proxy for anomaly score
            results["neural"] = float(torch.mean((tensor_in - recon) ** 2).item())
    except Exception as e:
        print(f"⚠️ VAE Pillar Error: {e}")

    # --- PILLAR 3: CLUSTERING (DBSCAN) ---
    try:
        dbscan = joblib.load("models/champion_dbscan.joblib")
        # Since DBSCAN doesn't 'predict', we calculate distance to the training set
        # For simplicity in this ensemble, we check if it was a noise point (-1)
        # or calculate a distance-based anomaly score
        core_samples = dbscan.components_
        dist = np.min(np.linalg.norm(core_samples - full_vector, axis=1))
        results["clustering"] = float(np.tanh(dist))  # Squish distance to 0-1 range
    except Exception as e:
        print(f"⚠️ DBSCAN Pillar Error: {e}")

    # Final Weighted Total (40/40/20)
    results["total"] = (results["supervised"] * 0.4) + \
                       (results["neural"] * 0.4) + \
                       (results["clustering"] * 0.2)

    return results


import re
import math


def sigmoid(x):
    # Subtracting 2 or 3 shifts the curve so a raw value of 1 doesn't jump to 0.7
    return 1 / (1 + math.exp(-(x - 3)))

def calibrate_score(raw_val, threshold=1.0):
    """
    Smoothly maps raw anomaly scores (MSE/Distance) to a 0-1 probability.
    Values below threshold stay low; values above scale logarithmically.
    """
    if raw_val <= threshold:
        # Map 0 -> threshold linearly to 0 -> 0.5
        # This gives the 'benefit of the doubt' to low-error transactions
        return (raw_val / threshold) * 0.5
    else:
        # Map values above threshold to 0.5 -> 1.0
        # math.log1p(x) is ln(1+x), great for squashing outliers
        scaled = 0.5 + (math.log1p(raw_val - threshold) / 5.0)
        return min(0.99, scaled) # Cap at 0.99 to keep it realistic


import re


def extract_detailed_scores(text: str, cc_last4: str = None):
    scores = {}
    mapping = {
        "supervised": r"(?:LR|Supervised)",
        "neural": r"(?:RNN|Neural|VAE)",
        "clustering": r"(?:DB|Clustering|DBSCAN)",
        "total": r"TOTAL"
    }

    for key, pattern in mapping.items():
        # Refined regex: matches numbers like 0.87, .87, or 87
        # Skips decorative dots or brackets
        regex = rf"{pattern}.*?(\d*\.?\d+)"
        match = re.search(regex, text, re.IGNORECASE)

        if match:
            try:
                val = match.group(1)
                scores[key] = float(val)
            except ValueError:
                # This specifically catches the '.' error you just saw
                raise ValueError(
                    f"❌ FLOAT CONVERSION ERROR: Found '{val}' for {key} in CC {cc_last4}. Text context: ...{text[-200:]}")
        else:
            raise ValueError(f"❌ MISSING SCORE: Could not find {key} in Agent output for CC {cc_last4}.")

    return scores

import os
from autogen_core.tools import FunctionTool


def save_report_to_disk(filename: str, content: str) -> str:
    """
    Saves a generated fraud report to the local 'reports' directory.
    The filename should end in .md.
    """
    directory = "mas_fraud_detector/reports"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return f"SUCCESS: Report saved to {filepath}"


# Wrap for AutoGen 0.4
publisher_tool = FunctionTool(
    save_report_to_disk,
    name="save_report_to_disk",
    description="Saves the final fraud report. Filename must include .md extension."
)
