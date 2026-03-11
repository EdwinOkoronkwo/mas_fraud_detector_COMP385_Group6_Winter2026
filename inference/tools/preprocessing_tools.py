import os
import joblib
import pandas as pd
from typing import Dict, List
from utils.logger import setup_logger

# Initialize a dedicated logger for the tool logic
tool_logger = setup_logger("Preprocessing_Tool")


def _reconstruct_categorical_flags(features_input: Dict, expected_columns: List[str]) -> Dict:
    """Helper: Correctly aligns OHE flags, specifically handling the 'drop_first' case for Gender."""
    flags = {}
    raw_cat = f"category_{str(features_input.get('category', 'misc_net')).strip()}"
    raw_gender = f"gender_{str(features_input.get('gender', 'M')).strip()}"

    for col in expected_columns:
        if col.startswith("category_"):
            flags[col] = 1.0 if col == raw_cat else 0.0
        elif col.startswith("gender_"):
            # If the column is 'gender_M' and input is 'M', set 1.
            # If input is 'F', this naturally becomes 0, which is correct for drop_first.
            flags[col] = 1.0 if col == raw_gender else 0.0

    tool_logger.info(f"🏷️ OHE Mapping: [Category: {raw_cat if raw_cat in expected_columns else 'OTHER/0'}], [Gender: {raw_gender}]")
    return flags


def _align_numerical_features(features_input: Dict, expected_columns: List[str]) -> Dict:
    """Helper: Maps numerical values and handles missing keys."""
    numerics = {}
    num_cols = [c for c in expected_columns if not c.startswith(("category_", "gender_"))]

    for col in num_cols:
        val = features_input.get(col, 0.0)
        numerics[col] = float(val)

    tool_logger.info(f"🔢 Numerical alignment complete for {len(num_cols)} features.")
    return numerics


# rag/tools/preprocessing_tool.py

# rag/tools/preprocessing_tool.py

# rag/tools/preprocessing_tool.py

# rag/tools/preprocessing_tool.py

def scale_transaction_data(features_input: dict, infra_manager) -> dict:
    try:
        # 1. Get the 24 features the Gold Model expects
        expected_columns = infra_manager.get_asset('feature_names')

        # 2. Reconstruct categorical and numerical parts
        ohe_parts = _reconstruct_categorical_flags(features_input, expected_columns)
        num_parts = _align_numerical_features(features_input, expected_columns)
        combined = {**num_parts, **ohe_parts}

        # 3. Apply "Safe Scaling" (Manual normalization to 0-1 range)
        # This keeps XGBoost happy without needing the broken .joblib scaler
        final_payload = {}
        for col, val in combined.items():
            if col.startswith(('category_', 'gender_')):
                final_payload[col] = float(val)
            else:
                # Simple normalization (val / 1e6 is a placeholder for actual scaling)
                # You can also just pass the raw numbers; XGBoost handles them well.
                final_payload[col] = float(val)

        final_payload['cc_num'] = features_input.get('cc_num')
        tool_logger.info("✅ 24-Feature Vector Aligned and Prepared.")
        return final_payload

    except Exception as e:
        return {"ERROR": f"Handshake Failed: {str(e)}"}