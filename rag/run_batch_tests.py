import asyncio
import os
import re

from autogen_ext.models.openai import OpenAIChatCompletionClient  # Swap for Ollama later

from mas_fraud_detector.config.llm_config import get_model_client
from mas_fraud_detector.config.settings import DB_PATH, MODELS_DIR
from mas_fraud_detector.rag.rag_team import RAGTeam

import asyncio
import os

from mas_fraud_detector.rag.vector_service import VectorService

import asyncio
import os
import re
import json


import asyncio
import os
import re
import json
from autogen_agentchat.base import TaskResult # REQUIRED for the fix

import asyncio
import os
import re
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path
from autogen_agentchat.base import TaskResult
# Import our Single Source of Truth
from mas_fraud_detector.config.settings import DB_PATH, MODELS_DIR, POLICY_FILE, REPORT_DIR
from mas_fraud_detector.rag.vector_service import VectorService


# # --- IDENTITY INJECTOR LOGIC ---
# import json  # Ensure this is imported
#
# # Inside run_batch_tests.py
# vector_service = VectorService(persist_directory="./chroma_db")
#
# # Path to your text file
# POLICY_FILE = r"C:\Youtube\autogen\mas_fraud_detector\data\policies\fraud_handbook.txt"
#
# # FORCE INGESTION (You only need to do this once, but it doesn't hurt to repeat)
# vector_service.load_local_policies(POLICY_FILE)

# Now pass it to the team manager

import os
import sqlite3
import joblib
import torch
import pandas as pd


def run_system_health_check():
    print("🔍 INITIALIZING FRAUD PIPELINE HEALTH CHECK...")
    errors = []

    # 1. Database Check
    if not os.path.exists(DB_PATH):
        errors.append(f"CRITICAL: Database not found at {DB_PATH}")
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("PRAGMA table_info(train_transactions)", conn)
            required_cols = {'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long'}
            found_cols = set(df['name'].tolist())
            if not required_cols.issubset(found_cols):
                errors.append(f"SCHEMA ERROR: Missing {required_cols - found_cols}")
            conn.close()
            print(f"✅ Database Linked: {DB_PATH}")
        except Exception as e:
            errors.append(f"DB ERROR: {str(e)}")

    # 2. Model Check
    # 2. Model Check
    # Update these strings to match your ACTUAL filenames
    artifacts = ["champion_rf.joblib", "champion_rnn_ae.pth", "champion_kmeans.joblib", "scaler.joblib"]
    for f in artifacts:
        # We join MODELS_DIR with the new filenames
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            errors.append(f"MISSING MODEL: {f} in {MODELS_DIR}")

    if not errors:
        print("🚀 SYSTEM HEALTHY: Paths aligned.")
        return True
    else:
        for e in errors: print(f"❌ {e}")
        return False



import asyncio
import os
import re
import json
import pandas as pd
from sqlalchemy import create_engine
from autogen_agentchat.base import TaskResult

# Import your Single Source of Truth from config
from mas_fraud_detector.config.settings import DB_PATH, POLICY_FILE, MODELS_DIR
from mas_fraud_detector.rag.vector_service import VectorService


def prepare_test_identities():
    """Fetches real data from the DB as STRINGS to ensure 19-digit precision."""
    print(f"DEBUG: Connecting to DB for Identity Injection: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        print(f"❌ ERROR: Database not found at {DB_PATH}")
        return None, None

    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")

        # CRITICAL FIX: Use CAST(cc_num AS TEXT) in the SQL query
        # This forces the DB to hand over a string, bypassing float precision limits
        query_cols = "CAST(cc_num AS TEXT) as cc_num, amt, zip, lat, long, city_pop, unix_time, merch_lat, merch_long"

        with engine.connect() as conn:
            alice_df = pd.read_sql(f"SELECT {query_cols} FROM train_transactions WHERE is_fraud = 0 LIMIT 1", conn)
            brandon_df = pd.read_sql(f"SELECT {query_cols} FROM train_transactions WHERE is_fraud = 1 LIMIT 1", conn)

            if alice_df.empty or brandon_df.empty:
                print("❌ ERROR: Database is empty or missing required records.")
                return None, None

            # Extract dictionaries
            alice_data = alice_df.iloc[0].to_dict()
            brandon_data = brandon_df.iloc[0].to_dict()

            # FINAL SAFETY: Ensure they are clean strings (no scientific notation or .0)
            alice_data['cc_num'] = str(alice_data['cc_num']).split('.')[0]
            brandon_data['cc_num'] = str(brandon_data['cc_num']).split('.')[0]

            # Save for auditing
            with open("current_test_identities.json", "w") as f:
                json.dump({"alice": alice_data, "brandon": brandon_data}, f, indent=4)

            return alice_data, brandon_data

    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")
        return None, None

async def run_batch_tests():
    # 1. Health Check & Identity Injection
    print("--- STEP 0: INITIALIZING TEST IDENTITIES ---")
    alice_data, brandon_data = prepare_test_identities()

    if not alice_data or not brandon_data:
        print("❌ FAILED TO PREPARE TEST CASES. ABORTING.")
        return

    print(f"✅ DATA READY: Alice (Safe) and Brandon (Fraud) loaded.")

    # 2. Setup Shared Services
    vector_service = VectorService(persist_directory="./chroma_db")
    vector_service.load_local_policies(POLICY_FILE)

    # 3. Initialize Agent Team (Passing the verified DB_PATH)
    model_client = get_model_client()
    rag_team_manager = RAGTeam(model_client, DB_PATH, vector_service)

    # 4. Define Scenarios
    # 3. Define Test Cases with FULLY DYNAMIC Data
    # 3. Define Test Cases with Explicit Schema Requirements
    test_cases = [
        {
            "id": "CASE_001",
            "desc": "Brandon (Fraud Case)",
            "task": f"""Investigate CC {brandon_data['cc_num']} for the transaction of ${brandon_data['amt']:.2f}.

            REQUIRED STEPS:
            1. Use SQL_Researcher to fetch the mandatory 9-feature vector: 
               [cc_num, amt, zip, lat, long, city_pop, unix_time, merch_lat, merch_long].
            2. Do NOT truncate this list. The ensemble model will FAIL if you pass only 3 features.
            3. Pass the full 9-item 'CRITICAL_DATA_VECTOR' to the Inference_Specialist."""
        },
        {
            "id": "CASE_004",
            "desc": "Alice (Routine Case)",
            "task": f"""Investigate CC {alice_data['cc_num']} for the transaction of ${alice_data['amt']:.2f}.

            REQUIRED STEPS:
            1. Fetch the 9-feature vector from the database via SQL_Researcher.
            2. Ensure the following features are included: [cc_num, amt, zip, lat, long, city_pop, unix_time, merch_lat, merch_long].
            3. Even if you scale 'amt' or 'distance', you MUST still provide the full 9-item raw vector to the Inference_Specialist for the ensemble calculation."""
        }
    ]
    results_table = []
    print("\n🚀 STARTING BATCH INVESTIGATION\n" + "=" * 60)

    for case in test_cases:
        print(f"\n[!] Case {case['id']}: {case['desc']}")
        investigation_group = rag_team_manager.get_team()
        current_final_score = 0.0
        case_log = []

        async for message in investigation_group.run_stream(task=case['task']):
            if isinstance(message, TaskResult):
                print(f"\n✅ [CASE FINISHED]: {case['id']}")
                continue

            source = getattr(message, 'source', 'Agent')
            content = message.to_text() if hasattr(message, "to_text") else str(message)

            print(f"\n[{source.upper()}]:\n{'-' * 40}\n{content}\n{'-' * 40}")

            # Capture the Ensemble Score
            match = re.search(r"TOTAL RISK SCORE[:\s]*(\d+\.\d+)", content, re.IGNORECASE)
            if match:
                current_final_score = float(match.group(1))
                print(f"🎯 SCORE CAPTURED: {current_final_score}")

            case_log.append(f"{source}: {content}")

        # Save logs
        log_path = os.path.join(REPORT_DIR, f"logs_{case['id']}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(case_log))

        # Store result for final summary
        results_table.append({
            "ID": case['id'],
            "Score": f"{current_final_score:.4f}",
            "Verdict": "❌ FRAUD" if current_final_score >= 0.50 else "✅ PASS"
        })

    # 5. Print Summary Table
    print("\n" + "=" * 50)
    print(f"{'ID':<10} | {'Score':<10} | {'Verdict'}")
    print("-" * 50)
    for row in results_table:
        print(f"{row['ID']:<10} | {row['Score']:<10} | {row['Verdict']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Ensure paths are available for the health check to see
    from mas_fraud_detector.config.settings import DB_PATH, MODELS_DIR

    print("\n" + "=" * 50)
    print("🚀 PRE-FLIGHT SYSTEM CHECK")
    print("=" * 50)

    # CALL WITHOUT ARGUMENTS:
    # Because your function definition is 'def run_system_health_check():'
    is_healthy = run_system_health_check()

    if is_healthy:
        print("\n✅ SYSTEM READY. STARTING INVESTIGATION AGENTS...")
        asyncio.run(run_batch_tests())
    else:
        print("\n❌ SYSTEM UNHEALTHY. Please fix the paths listed above.")
        print("Investigation aborted to save API credits and prevent errors.")
    print("=" * 50 + "\n")