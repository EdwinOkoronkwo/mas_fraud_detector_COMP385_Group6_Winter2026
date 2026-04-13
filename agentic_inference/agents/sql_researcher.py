import json
import sqlite3

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

import sqlite3
import json
import pandas as pd



import sqlite3
import json
import pandas as pd


class SQLResearcher:
    def __init__(self, model_client, db_path):
        """
        Initializes the SQL Researcher agent with the smart query tool.
        """
        self.db_path = db_path

        # 1. Define the tool using the instance method.
        # FunctionTool automatically handles 'self' when passed self.method_name.
        query_tool = FunctionTool(
            self.run_smart_transaction_query,
            name="run_smart_transaction_query",
            description=(
                "Lookup transactions in the database. Arguments: cc_num (16-digit or tail), "
                "amt, merchant, first, last. Uses cascaded matching for high reliability."
            )
        )

        # 2. Initialize the AssistantAgent
        self.agent = AssistantAgent(
            name="SQL_Researcher",
            model_client=model_client,
            tools=[query_tool],
            system_message="""You are the Data Custodian. 
            Your primary goal is to GROUND the audit in real database records.
            1. Use the FULL 16-digit IDENTIFIER provided in the prompt for the 'cc_num' argument. 
            2. Do NOT truncate or shorten the number unless specifically instructed.
            3. Use the merchant name and first/last names exactly as provided in the prompt.
            4. If a search returns DATA_MISSING, do not hallucinate; report the missing data."""
        )

    def run_smart_transaction_query(self, cc_num: str, amt: float, merchant: str = None,
                                    first: str = None, last: str = None) -> str:
        """
        The 'once and for all' search tool. Handles prefixing, numeric rounding,
        and identifier wildcards.
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # --- 1. PRE-PROCESSING ---
            # CC Wildcard: Ensures '9375' matches '...9375'
            clean_cc = "".join(filter(str.isdigit, str(cc_num)))
            cc_pattern = f"%{clean_cc}%"

            # Merchant Wildcard: Handles 'fraud_Pacocha-Bauch' vs 'Pacocha-Bauch'
            clean_m = merchant.replace('fraud_', '').strip() if merchant else ""
            m_pattern = f"%{clean_m}%"

            # Amount Buffer: Handles float precision differences between ML and SQL
            low_amt, high_amt = float(amt) - 0.05, float(amt) + 0.05

            # --- 2. EXECUTION ---
            # Search Strategy: Name/CC match + Merchant + Amount
            query = """
                    SELECT * FROM test_transactions
                    WHERE (first LIKE ? OR CAST(cc_num AS TEXT) LIKE ?)
                      AND merchant LIKE ?
                      AND amt BETWEEN ? AND ? 
                    LIMIT 1
                    """
            params = [f"{first}%", cc_pattern, m_pattern, low_amt, high_amt]
            df = pd.read_sql_query(query, conn, params=params)

            conn.close()

            if df.empty:
                return "DATA_MISSING: No transaction found matching these details."

            # Return the record as JSON for the Agent to process
            return f"CRITICAL_DATA_JSON: {json.dumps(df.iloc[0].to_dict(), default=str)}"

        except Exception as e:
            return f"[ERROR]: {str(e)}"