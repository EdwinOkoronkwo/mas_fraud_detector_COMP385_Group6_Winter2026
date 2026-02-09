import joblib
import torch
import numpy as np
import os

from mas_fraud_detector.rag.tools.rag_tools import scale_transaction_data, run_rf_prediction, run_rnn_prediction, \
    run_dbscan_prediction


# Import your functions from your tools file



def test_inference_pipeline():
    print("--- STARTING OFFLINE MODEL TEST ---")

    # Raw Data for Brandon Pittman
    raw_amt, raw_hr, raw_dist = 88.93, 6, 57.6

    try:
        # 1. Test Scaling
        scaled = scale_transaction_data(raw_amt, raw_hr, raw_dist)
        s_a, s_h, s_d = scaled["s_amount"], scaled["s_hour"], scaled["s_dist"]
        print(f"✅ Scaling Success: {scaled}")

        # 2. Test RF
        rf = run_rf_prediction(s_a, s_h, s_d)
        print(f"✅ RF Result: {rf}")

        # 3. Test RNN
        rnn = run_rnn_prediction(s_a, s_h, s_d)
        print(f"✅ RNN Result: {rnn}")

        # 4. Test DBSCAN
        db = run_dbscan_prediction(s_a, s_h, s_d)
        print(f"✅ DBSCAN Result: {db}")

    except Exception as e:
        print(f"❌ PIPELINE ERROR: {str(e)}")


if __name__ == "__main__":
    test_inference_pipeline()