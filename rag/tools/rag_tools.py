import json
import re
import sqlite3
from langchain.tools import tool
from autogen_ext.tools.langchain import LangChainToolAdapter

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


def scale_transaction_data(amount: float, hour: int, dist: float) -> dict:
    """
    Handles the 9-feature requirement of the saved scaler.
    Assumes amount, hour, and dist are the first three features.
    """
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    # 1. Create a template of 9 zeros (or the mean of your training data)
    # This matches the shape the scaler expects
    full_feature_row = np.zeros((1, 9))

    # 2. Assign your 3 active features to their original positions
    # (Update these indices [0, 1, 2] if they were in different columns in your CSV)
    full_feature_row[0, 0] = amount
    full_feature_row[0, 1] = hour
    full_feature_row[0, 2] = dist

    # 3. Scale the full row
    scaled_full = scaler.transform(full_feature_row)[0]

    # 4. Extract only the 3 scaled values we need for our models
    return {
        "s_amount": float(scaled_full[0]),
        "s_hour": float(scaled_full[1]),
        "s_dist": float(scaled_full[2])
    }


def run_rf_prediction(features: list) -> str:
    model = joblib.load(os.path.join(MODEL_DIR, "champion_rf.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    # 1. Convert list to numpy array
    raw_data = np.array(features).reshape(1, -1)

    # 2. SCALE the data (Crucial! Raw zip/lat will break the model)
    scaled_data = scaler.transform(raw_data)

    # 3. Predict
    prob = float(model.predict_proba(scaled_data)[0][1])
    return f"RF_SCORE: {prob:.4f}"

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

# 2. Update the Prediction Function
def run_rnn_prediction(features: list) -> str:
    """Neural Pillar (40%): Calibrated Residual LSTM-AE."""
    MODEL_PATH = os.path.join(MODEL_DIR, "champion_rnn_ae.pth")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

    try:
        model = Residual_LSTM_AE(in_dim=9)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()

        scaler = joblib.load(SCALER_PATH)
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            reconstruction = model(input_tensor)
            mse_value = torch.mean((input_tensor - reconstruction) ** 2).item()

        # CALIBRATION: Increased to 0.15 to stop over-flagging normal noise
        training_threshold = 0.15
        prob = 1.0 - np.exp(-mse_value / training_threshold)

        return f"RNN_SCORE: {prob:.4f} (MSE: {mse_value:.6f})"
    except Exception as e:
        return f"RNN_PREDICTION_ERROR: {str(e)}"


def run_dbscan_prediction(features: list) -> str:
    """Clustering Pillar (20%): DBSCAN Distance in 9D Space"""
    model = joblib.load(os.path.join(MODEL_DIR, "champion_dbscan.joblib"))
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

    # 1. Scale the input
    scaler = joblib.load(SCALER_PATH)
    raw_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(raw_array)

    # 2. Calculate distance to the nearest "Normal" cluster core point
    from sklearn.neighbors import NearestNeighbors
    # We fit the components of the trained DBSCAN model
    nn = NearestNeighbors(n_neighbors=1).fit(model.components_)
    distances, _ = nn.kneighbors(scaled_features)
    dist = distances[0][0]

    # Normalize distance to a 0.0 - 1.0 score
    # Change this line in run_dbscan_prediction:
    sensitivity_factor = 0.5
    actual_score = 1.0 - np.exp(-dist / (model.eps * sensitivity_factor))
    return f"DBSCAN_SCORE: {actual_score:.4f}"

def run_lr_prediction(features: list) -> str:
    """Logistic Regression prediction for the Supervised Pillar."""
    try:
        # Load the Logistic Regression champion and the scaler
        model = joblib.load(os.path.join(MODEL_DIR, "champion_lr.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

        # 1. Convert list to numpy array and reshape for a single sample
        raw_data = np.array(features).reshape(1, -1)

        # 2. Scale the data (Consistency with training is key)
        scaled_data = scaler.transform(raw_data)

        # 3. Predict probability of class 1 (Fraud)
        prob = float(model.predict_proba(scaled_data)[0][1])
        return f"LR_SCORE: {prob:.4f}"
    except Exception as e:
        return f"LR_ERROR: {str(e)}"

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
    description="Saves the final fraud report to a local markdown file."
)

#
# def calculate_model_based_ensemble(s_amount: float, s_hour: float, s_dist: float) -> str:
#     """
#     Executes the 40/40/20 ensemble (LR, RNN, DBSCAN) and returns a weighted score.
#     """
#     # Create the feature list expected by the models
#     features = [s_amount, s_hour, s_dist]
#
#     # 1. Get raw outputs from your "Best in Class" winners
#     lr_res = run_lr_prediction(features)
#     rnn_res = run_rnn_prediction(features)
#     db_res = run_dbscan_prediction(features)
#
#     # 2. Extract numeric scores with error handling
#     try:
#         # Parse "LR_SCORE: 0.8333"
#         lr_score = float(lr_res.split(": ")[1])
#
#         # Parse "RNN_SCORE: 0.3885 (MSE)" -> extract 0.3885
#         # Note: In your strategy, RNN score is usually (1 - MSE) or a normalized anomaly score
#         rnn_raw_mse = float(rnn_res.split(": ")[1].split(" ")[0])
#         # Normalizing RNN: Lower MSE = Lower Fraud. We use a simple inversion or raw MSE depending on your RNN tool's output
#         rnn_score = rnn_raw_mse
#
#         # Parse "DBSCAN_SCORE: 1.0" (1 for anomaly, 0 for normal)
#         db_score = float(db_res.split(": ")[1])
#
#     except Exception as e:
#         return f"Ensemble Error: Failed to parse model outputs. {str(e)}"
#
#     # 3. Apply the 40/40/20 Weighted Ensemble logic
#     # LR (40%) + RNN (40%) + DBSCAN (20%)
#     final_score = (lr_score * 0.4) + (rnn_score * 0.4) + (db_score * 0.2)
#
#     # Determine Risk Status
#     status = "CRITICAL" if final_score > 0.8 else "HIGH" if final_score > 0.5 else "LOW"
#
#     report = (
#         f"--- FINAL BEST-IN-CLASS ENSEMBLE REPORT ---\n"
#         f"Supervised (Logistic Regression): {lr_score:.4f}\n"
#         f"Neural (RNN Autoencoder MSE):    {rnn_score:.4f}\n"
#         f"Clustering (DBSCAN Outlier):     {db_score:.4f}\n"
#         f"-------------------------------------------\n"
#         f"COMBINED ENSEMBLE SCORE:         {final_score:.4f}\n"
#         f"RISK LEVEL:                      {status}\n"
#     )
#
#     # Log to your MAS system
#     if 'log_inference_event' in globals():
#         log_inference_event("FINAL_ENSEMBLE_AGGREGATION", {"final_score": final_score, "status": status})
#
#     return report







from typing import List

# This function is now the 'Core Logic'.
# The Pydantic EnsembleInput class handles the 'Alice Fix' and validation before this runs.

def execute_champion_ensemble(features: list) -> dict:
    """
    CLEANED: Unified entry point for raw model execution only.
    Returns a dictionary of raw scores for the Gatekeeper to process.
    """
    try:
        # DATA INTEGRITY: Clean the CC float representation
        features[0] = int(round(float(features[0])))
        amt = float(features[1])

        # RUN PREDICTIONS (Parallel in logic, sequential in execution)
        lr_res = run_lr_prediction(features)
        rnn_res = run_rnn_prediction(features)
        db_res = run_dbscan_prediction(features)

        # PARSE SCORES
        def _parse(res):
            return float(res.split(": ")[1].split(" ")[0])

        return {
            "lr": _parse(lr_res),
            "rnn": _parse(rnn_res),
            "db": _parse(db_res),
            "amt": amt,
            "cc_num": str(features[0])
        }

    except Exception as e:
        print(f"❌ EXECUTION ERROR: {str(e)}")
        return {"lr": 0.0, "rnn": 0.0, "db": 0.0, "amt": 0.0, "error": str(e)}

# def execute_champion_ensemble(features: list) -> str:
#     """
#     Unified entry point for LR, RNN, and DBSCAN.
#     Updated with Adaptive High-Value Weighting (60/20/20) and Screamer Logic.
#     """
#     try:
#         # 1. DATA INTEGRITY: Clean the CC float representation
#         raw_cc_float = float(features[0])
#         clean_cc_int = int(round(raw_cc_float))
#         features[0] = clean_cc_int
#         clean_cc_str = str(clean_cc_int)
#
#         amt = float(features[1])
#
#         # 2. RUN PREDICTIONS
#         lr_res = run_lr_prediction(features)
#         rnn_res = run_rnn_prediction(features)
#         db_res = run_dbscan_prediction(features)
#
#         # 3. PARSE SCORES
#         def _parse(res):
#             return float(res.split(": ")[1].split(" ")[0])
#
#         lr_s = _parse(lr_res)
#         rnn_s = _parse(rnn_res)
#         db_s = _parse(db_res)
#
#         # 4. ADAPTIVE ENSEMBLE LOGIC
#         # Threshold-based weighting shift
#         if amt > 1000:
#             # High-Value Mode: Prioritize Sequence (RNN)
#             weighted_avg = (rnn_s * 0.6) + (lr_s * 0.2) + (db_s * 0.2)
#             mode_label = "HIGH-VALUE (RNN PRIORITIZED)"
#         else:
#             # Standard Mode: 40/40/20
#             weighted_avg = (rnn_s * 0.4) + (lr_s * 0.4) + (db_s * 0.2)
#             mode_label = "STANDARD"
#
#         highest_score = max(lr_s, rnn_s, db_s)
#
#         # RULE 1: THE SCREAMER (If any model is > 0.90 sure, don't let the average dilute it)
#         if highest_score > 0.90:
#             final_score = max(weighted_avg, highest_score * 0.95)
#
#         # RULE 2: THE CONSENSUS (Mid-tier fraud protection)
#         elif amt > 200 and sum(1 for s in [lr_s, rnn_s, db_s] if s > 0.35) >= 2:
#             final_score = max(weighted_avg, 0.51)
#
#         else:
#             final_score = weighted_avg
#
#         # 5. STRUCTURED OUTPUT
#         output_for_agent = f"""TOTAL RISK SCORE: {final_score:.4f}
# CALCULATION MODE: {mode_label}
#
# DETAILED MODEL BREAKDOWN:
# - Supervised (LR): {lr_s:.4f}
# - Neural (RNN): {rnn_s:.4f}
# - Clustering (DBSCAN): {db_s:.4f}
#
# INPUT DATA AUDIT:
# - CC_NUM: {clean_cc_str}
# - AMOUNT: ${amt:.2f}
# - UNIX_TIME: {int(features[6])}
#
# FINAL_SCORES: LR: {lr_s:.4f}, RNN: {rnn_s:.4f}, DB: {db_s:.4f}, TOTAL: {final_score:.4f}
# """
#         print(f"✅ CALCULATION UPDATED [{mode_label}]: {final_score:.4f}")
#         return output_for_agent
#
#     except Exception as e:
#         print(f"❌ ENSEMBLE ERROR: {str(e)}")
#         return f"TOTAL RISK SCORE: 0.9999 (INFERENCE_FAILED: {str(e)})"





import re

def extract_detailed_scores(text):
    # Initialize with 'lr' as the primary supervised key
    scores = {"lr": 0.0, "rnn": 0.0, "db": 0.0, "Total": 0.0}

    # 1. PRIORITY: Enhanced Footer Match
    # This pattern is now 'label-blind' for the first score to handle RF/LR/Supervised mixups
    footer = re.search(
        r"FINAL_SCORES:.*?(?:RF|LR|Supervised):?\s*([\d.]+).*?RNN:?\s*([\d.]+).*?DB:?\s*([\d.]+).*?TOTAL:?\s*([\d.]+)",
        text, re.I | re.S
    )

    if footer:
        # We map the first capture group (RF or LR) into our 'lr' key
        scores["lr"], scores["rnn"], scores["db"], scores["Total"] = map(float, footer.groups())
        print(f"DEBUG: ✅ Footer Match (Label Agnostic): {scores}")
        return scores

    # 2. FALLBACK: Scavenger with Dual-Label Support
    patterns = {
        "lr": r"(?:LR|RF|Logistic|Supervised).*?(0\.\d{1,4}|1\.0+)", # Catches both
        "rnn": r"(?:RNN|Neural|Sequence).*?(0\.\d{1,4}|1\.0+)",
        "db": r"(?:DB|DBSCAN|Clustering).*?(0\.\d{1,4}|1\.0+)",
        "Total": r"(?:TOTAL|AI Score|Final Risk Score).*?(0\.\d{1,4}|1\.0+)"
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            scores[key] = float(m.group(1))

    print(f"DEBUG: 🔍 Scavenger Match (Updated Filter): {scores}")
    return scores


def execute_unified_ensemble(features: list, is_simplified: bool = False) -> str:
    """
    Consolidated Fraud Engine.
    Handles both the full 9-feature vector and simplified 3-feature sets.
    Applies the 'Screamer Rule' and 'High-Value Consensus' logic.
    """
    try:
        # 1. DATA INTEGRITY & CLEANING
        # If it's the full vector, clean the CC float representation
        if len(features) >= 9:
            raw_cc_float = float(features[0])
            clean_cc_int = int(round(raw_cc_float))
            features[0] = clean_cc_int
            cc_display = str(clean_cc_int)[-4:]  # Last 4 for reporting
        else:
            cc_display = "INTERNAL"

        amt = float(features[1]) if len(features) > 1 else float(features[0])

        # 2. UNIFIED INFERENCE
        lr_res = run_lr_prediction(features)
        rnn_res = run_rnn_prediction(features)
        db_res = run_dbscan_prediction(features)

        # 3. ROBUST PARSING
        def parse_score(res):
            # Handles "Label: 0.XXXX" or "Label: 0.XXXX (Details)"
            return float(res.split(": ")[1].split(" ")[0])

        lr_s = parse_score(lr_res)
        rnn_s = parse_score(rnn_res)
        db_s = parse_score(db_res)

        # 4. ENSEMBLE LOGIC (The Screamer & Consensus Rules)
        weighted_avg = (rnn_s * 0.4) + (lr_s * 0.4) + (db_s * 0.2)
        highest_score = max(lr_s, rnn_s, db_s)

        # Rule A: The Screamer Rule (One model is certain)
        if highest_score > 0.90:
            final_score = max(weighted_avg, highest_score * 0.95)

        # Rule B: The Consensus Rule (Multiple models are suspicious on mid-to-high value)
        elif amt > 200 and sum(1 for s in [lr_s, rnn_s, db_s] if s > 0.35) >= 2:
            final_score = max(weighted_avg, 0.51)

        else:
            final_score = weighted_avg

        # 5. GENERATE UNIFIED REPORT
        status = "CRITICAL" if final_score > 0.8 else "HIGH" if final_score > 0.5 else "LOW"

        output = f"""TOTAL RISK SCORE: {final_score:.4f}
RISK STATUS: {status}

DETAILED MODEL BREAKDOWN:
- Supervised (LR): {lr_s:.4f}
- Neural (RNN): {rnn_s:.4f}
- Clustering (DBSCAN): {db_s:.4f}

FINAL_SCORES: LR: {lr_s:.4f}, RNN: {rnn_s:.4f}, DB: {db_s:.4f}, TOTAL: {final_score:.4f}
"""
        print(f"✅ UNIFIED INFERENCE COMPLETE: {final_score:.4f}")
        return output

    except Exception as e:
        print(f"❌ ENSEMBLE ERROR: {str(e)}")
        return f"TOTAL RISK SCORE: 0.9999 (INFERENCE_FAILED: {str(e)})"