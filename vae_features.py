import os
import torch
import time
from datetime import datetime

model_path = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\models\champion_vae.pth"


def perform_deep_audit(path):
    if not os.path.exists(path):
        print(f"❌ ERROR: File not found at {path}")
        return

    # 1. TIMESTAMP CHECK
    m_time = os.path.getmtime(path)
    dt_object = datetime.fromtimestamp(m_time)
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    # 2. FEATURE COUNT CHECK (The Weights)
    try:
        state_dict = torch.load(path, map_location='cpu')
        # encoder.0.weight shape is [neurons, features]
        input_layer_shape = state_dict['encoder.0.weight'].shape
        feature_count = input_layer_shape[1]
    except Exception as e:
        print(f"❌ ERROR reading internal weights: {e}")
        return

    print("====================================================")
    print("🔎 VAE FILE SYSTEM & ARCHITECTURE AUDIT")
    print("====================================================")
    print(f"📍 PATH:     {path}")
    print(f"📅 MODIFIED: {formatted_time}")
    print(f"📊 FEATURES: {feature_count} Input Neurons")
    print("----------------------------------------------------")

    # Final Verdict
    if feature_count == 27:
        print("✅ STATUS: COMPATIBLE. (Matches Phase 1 Enhanced Mode)")
    else:
        print(f"⚠️ STATUS: DISCONNECTED. (Expected 27, found {feature_count})")

    if dt_object.date() == datetime.now().date():
        print("🕒 RECENCY: This file was created/updated TODAY.")
    else:
        print("🕒 RECENCY: This is an OLD file from a previous session.")
    print("====================================================")


perform_deep_audit(model_path)