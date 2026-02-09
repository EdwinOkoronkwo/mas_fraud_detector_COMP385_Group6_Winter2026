import pandas as pd
from sqlalchemy import create_engine
import os

import pandas as pd
from sqlalchemy import create_engine
import os
import json

# Run this to see why the search failed
import sqlite3
import pandas as pd
from mas_fraud_detector.config.settings import DB_PATH

conn = sqlite3.connect(DB_PATH)
# Search ONLY by the first 6 digits of the CC to find the correct record
df = pd.read_sql_query("SELECT cc_num, amt FROM train_transactions WHERE cc_num LIKE '466631%' LIMIT 5", conn)
print(df)
conn.close()


def test_and_log_data(db_path):
    print(f"--- 🔍 STARTING DATABASE DIAGNOSTIC ---")

    if not os.path.exists(db_path):
        print(f"❌ ERROR: Database not found at {db_path}")
        return

    engine = create_engine(f"sqlite:///{db_path}")
    cols = ["cc_num", "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
    cols_str = ", ".join(cols)

    try:
        with engine.connect() as conn:
            print("📡 Connection successful. Fetching rows...")

            # Fetch Normal and Fraud rows
            alice_df = pd.read_sql(f"SELECT {cols_str} FROM train_transactions WHERE is_fraud = 0 LIMIT 1", conn)
            brandon_df = pd.read_sql(f"SELECT {cols_str} FROM train_transactions WHERE is_fraud = 1 LIMIT 1", conn)

            if alice_df.empty or brandon_df.empty:
                print("❌ ERROR: One or both queries returned no data. Check table name/content.")
                return

            # Convert to dict
            alice_data = alice_df.iloc[0].to_dict()
            brandon_data = brandon_df.iloc[0].to_dict()

            # --- THE FIX: CASTING TO PREVENT SCIENTIFIC NOTATION ---
            # We convert CC and Zip to int, others to float
            for data in [alice_data, brandon_data]:
                data['cc_num'] = int(data['cc_num'])
                data['zip'] = int(data['zip'])
                # All other features ensure they are floats
                for key in ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']:
                    data[key] = float(data[key])

            # --- VERBOSE LOGGING ---
            for label, data in [("ALICE (NORMAL)", alice_data), ("BRANDON (FRAUD)", brandon_data)]:
                print(f"\n✅ {label} PROFILE:")
                print("-" * 30)
                for key, val in data.items():
                    print(f"{key:<12}: {val}")
                print("-" * 30)

            return alice_data, brandon_data

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return None, None


# Run the test
DB_PATH = r"C:\Youtube\autogen\mas_fraud_detector\data\database.sqlite"
alice, brandon = test_and_log_data(DB_PATH)

if alice and brandon:
    print("\n🚀 DATA TEST PASSED. You can now safely integrate this into run_batch_tests.py.")