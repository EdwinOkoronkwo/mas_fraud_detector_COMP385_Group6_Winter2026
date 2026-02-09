from langchain.tools import tool
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import SQLDatabase
from autogen_agentchat.agents import AssistantAgent

from autogen_core.tools import FunctionTool # <--- Native AutoGen tool class
from autogen_agentchat.agents import AssistantAgent

import sqlite3
import pandas as pd

import sqlite3
import pandas as pd
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool


class SQLResearcher:
    def __init__(self, model_client, db_path):
        self.db_path = db_path

        # Define the tool AS a function first
        def run_smart_transaction_query(cc_num: str, amt: float) -> str:
            try:
                conn = sqlite3.connect(self.db_path)

                # 1. CLEAN THE INPUT: Ensure CC is a string and remove any decimal points
                # This prevents '4.66e18' or '4666...2944.0' from breaking the search
                clean_cc = str(cc_num).split('.')[0]

                # 2. STRING-SAFE SQL: Use CAST to compare text-to-text
                query = """
                        SELECT CAST(cc_num AS TEXT) as cc_num,
                               amt, \
                               zip, \
                               lat, \
                               long, \
                               city_pop,
                               unix_time, \
                               merch_lat, \
                               merch_long
                        FROM train_transactions
                        WHERE CAST(cc_num AS TEXT) = ?
                          AND amt BETWEEN ? AND ? LIMIT 1
                        """
                params = [clean_cc, amt - 0.05, amt + 0.05]

                df = pd.read_sql_query(query, conn, params=params)
                conn.close()

                if df.empty:
                    return f"[DATA_UNAVAILABLE]: No record found for CC {clean_cc} at ${amt:.2f}."

                # 3. EXTRACT AS STRINGS/FLOATS
                row = df.iloc[0]
                feature_vector = [
                    str(row['cc_num']),  # Keep CC as String to preserve the 19 digits
                    float(row['amt']),
                    int(row['zip']),
                    float(row['lat']),
                    float(row['long']),
                    int(row['city_pop']),
                    int(row['unix_time']),
                    float(row['merch_lat']),
                    float(row['merch_long'])
                ]

                # Pack for the Agent
                vector_json = json.dumps(feature_vector)
                return f"CRITICAL_DATA_VECTOR: {vector_json} | (Note: Pass this EXACT 9-item list to Inference_Specialist.)"

            except Exception as e:
                return f"[ERROR]: {str(e)}"

        # 1. Create the Tool
        query_tool = FunctionTool(
            run_smart_transaction_query,
            name="run_smart_transaction_query",
            description="Fetches the mandatory 9-feature vector required for fraud inference."
        )

        # 2. Create the Agent
        self.agent = AssistantAgent(
            name="SQL_Researcher",
            model_client=model_client,
            tools=[query_tool],
            system_message="""You are the Data Custodian. 
            Your ONLY job is to find the 9-feature vector and hand it over.

            1. Use 'run_smart_transaction_query' with the CC and Amount provided.
            2. When you get 'CRITICAL_DATA_VECTOR', repeat it EXACTLY. 
            3. Do not omit numbers. Do not scale numbers. 
            4. If the vector is not 9 items, you have failed your mission."""
        )