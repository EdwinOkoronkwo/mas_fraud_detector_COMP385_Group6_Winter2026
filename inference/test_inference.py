import pandas as pd
import joblib
import json
import os
import sqlite3
import xgboost as xgb
from config.settings import DB_PATH, PROJECT_ROOT

# PATHS
BASELINE_DIR = os.path.join(PROJECT_ROOT, "results", "baseline_experiment")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baselines", "manual_xgb_baseline.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")


def run_honest_evaluation():
    print(f"⚖️ EVALUATING ON UNSEEN TEST DATA (50/50 SPLIT)")

    # 1. Load Model, Scaler, and Config
    with open(os.path.join(BASELINE_DIR, "inference_config.json"), 'r') as f:
        config = json.load(f)
    expected_features = config["inference_params"]["expected_features"]

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    numeric_features = scaler.feature_names_in_

    # 2. Fetch UNSEEN Balanced Data from test_transactions
    conn = sqlite3.connect(DB_PATH)
    # Pulling from test_transactions ensures the model has NEVER seen this data
    f_query = "SELECT * FROM test_transactions WHERE is_fraud = 1 LIMIT 25"
    n_query = "SELECT * FROM test_transactions WHERE is_fraud = 0 LIMIT 25"

    fraud_df = pd.read_sql(f_query, conn)
    normal_df = pd.read_sql(n_query, conn)
    conn.close()

    test_df = pd.concat([fraud_df, normal_df]).sample(frac=1).reset_index(drop=True)
    print(f"✅ Unseen Dataset Prepared: {len(test_df)} samples.\n")

    # 3. Counters & Loop
    tp, tn, fp, fn = 0, 0, 0, 0
    print(f"{'ID':<5} | {'ACTUAL':<8} | {'PROB':<10} | {'RESULT'}")
    print("-" * 50)

    for i, row in test_df.iterrows():
        actual = int(row['is_fraud'])

        # --- THE PRE-PROCESSING BRIDGE ---
        # A. Scale numeric features
        numeric_input = [float(row.get(f, 0)) for f in numeric_features]
        scaled_values = scaler.transform([numeric_input])[0]
        scaled_dict = dict(zip(numeric_features, scaled_values))

        # B. One-Hot Encode categories (amt, zip, category_..., gender_...)
        row_df = pd.DataFrame([row])
        encoded_df = pd.get_dummies(row_df)

        # C. Merge scaled numbers back into the encoded row
        for col, val in scaled_dict.items():
            encoded_df[col] = val

        # D. Align to the 21 features the model expects
        final_input = encoded_df.reindex(columns=expected_features, fill_value=0)

        # 4. Predict
        dmatrix = xgb.DMatrix(final_input, feature_names=expected_features)
        prob = float(model.get_booster().predict(dmatrix)[0])
        prediction = 1 if prob > 0.5 else 0

        # Metrics logic
        if prediction == 1 and actual == 1:
            tp += 1; res = "✅ TP"
        elif prediction == 0 and actual == 0:
            tn += 1; res = "✅ TN"
        elif prediction == 1 and actual == 0:
            fp += 1; res = "⚠️ FP"
        else:
            fn += 1; res = "❌ FN"

        print(f"{i:<5} | {actual:<8} | {prob:.4f}     | {res}")

    # 5. Summary
    accuracy = (tp + tn) / len(test_df)
    print(f"\n📊 FINAL HONEST ACCURACY: {accuracy:.2%}")


if __name__ == "__main__":
    run_honest_evaluation()