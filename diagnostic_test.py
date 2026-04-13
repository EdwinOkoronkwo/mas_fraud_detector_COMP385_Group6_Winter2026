import sqlite3
import pandas as pd

import sqlite3
import pandas as pd
import json

# Path to your database
DB_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\data\database.sqlite"


def simulate_sql_researcher_tool(cc_num, amt, merchant, first, last):
    """
    Simulates the EXACT logic now inside your SQLResearcher class.
    """
    print(f"\n--- SIMULATING AGENT SEARCH ---")
    print(f"Agent is looking for: {first} {last} | ${amt} at {merchant}")

    try:
        conn = sqlite3.connect(DB_PATH)
        low_amt, high_amt = amt - 0.01, amt + 0.01
        df = pd.DataFrame()

        # 🚀 STRATEGY 1: The 'Jeffrey' Proof Logic
        m_pattern = f"%{merchant.replace('fraud_', '')}%"
        f_pattern = f"{first}%"  # This catches 'Jeffrey' when input is 'Jeff'

        print(f"Executing SQL with First Name Pattern: '{f_pattern}'")

        query = """
                SELECT * \
                FROM test_transactions
                WHERE first LIKE ? \
                  AND merchant LIKE ?
                  AND amt BETWEEN ? \
                  AND ? LIMIT 1 \
                """
        params = [f_pattern, m_pattern, low_amt, high_amt]

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            print("❌ SIMULATION FAILED: SQL returned no rows.")
        else:
            print("✅ SIMULATION SUCCESS: Record Found!")
            res = df.iloc[0].to_dict()
            print(f"Match Found: {res['first']} {res['last']} at {res['merchant']}")
            print(f"Full JSON Payload: {json.dumps(res, indent=2, default=str)[:200]}...")

    except Exception as e:
        print(f"🔥 ERROR during simulation: {e}")


# --- TEST CASE: The 'Jeff' Scenario ---
simulate_sql_researcher_tool(
    cc_num="2291163933867244",
    amt=2.86,
    merchant="Weimann, Kuhic and Beahan",  # Note: use the merchant found in your diagnostic
    first="Jeff",
    last="Unknown"
)