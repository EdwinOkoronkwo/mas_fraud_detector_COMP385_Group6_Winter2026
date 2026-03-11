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

        def run_smart_transaction_query(cc_num: str, amt: float) -> str:
            try:
                conn = sqlite3.connect(self.db_path)
                # FIX 1: Query the 'test_transactions' table for scientific validity
                # SELECT * ensures we get 'category' and 'gender' for the 21-feature vector
                query = "SELECT * FROM test_transactions WHERE cc_num = ? AND amt = ? LIMIT 1"
                df = pd.read_sql_query(query, conn, params=[cc_num, amt])
                conn.close()

                if df.empty: return "[ERROR]: No record found in test set."

                # Convert the first row to a dictionary
                data_dict = df.iloc[0].to_dict()

                # Ensure we cast cc_num to string for the JSON
                data_dict['cc_num'] = str(data_dict['cc_num'])

                return f"CRITICAL_DATA_JSON: {json.dumps(data_dict)}"
            except Exception as e:
                return f"[ERROR]: {str(e)}"

        query_tool = FunctionTool(
            run_smart_transaction_query,
            name="run_smart_transaction_query",
            description="Fetches the full transaction record (including category) from the test database."
        )

        self.agent = AssistantAgent(
            name="SQL_Researcher",
            model_client=model_client,
            tools=[query_tool],
            system_message="""You are the Data Custodian. 
            Your ONLY job is to retrieve the full transaction record and hand it over.

            1. Use 'run_smart_transaction_query' with the CC and Amount provided.
            2. When you receive 'CRITICAL_DATA_JSON', repeat the entire JSON block EXACTLY.
            3. Do NOT summarize or omit fields like 'category' or 'gender'.
            4. If the record is missing, report that the transaction does not exist in the test database."""
        )

# class SQLResearcher:
#     def __init__(self, model_client, db_path):
#         self.db_path = db_path
#
#         # Define the tool AS a function first
#         def run_smart_transaction_query(cc_num: str, amt: float) -> str:
#             try:
#                 conn = sqlite3.connect(self.db_path)
#                 # Use pandas to get a dictionary directly
#                 query = "SELECT * FROM train_transactions WHERE cc_num = ? AND amt = ? LIMIT 1"
#                 df = pd.read_sql_query(query, conn, params=[cc_num, amt])
#                 conn.close()
#
#                 if df.empty: return "[ERROR]: No record found."
#
#                 # Convert the first row to a dictionary
#                 data_dict = df.iloc[0].to_dict()
#                 return f"CRITICAL_DATA_JSON: {json.dumps(data_dict)}"
#             except Exception as e:
#                 return f"[ERROR]: {str(e)}"
#         # 1. Create the Tool
#         query_tool = FunctionTool(
#             run_smart_transaction_query,
#             name="run_smart_transaction_query",
#             description="Fetches the mandatory 9-feature vector required for fraud inference."
#         )
#
#         # 2. Create the Agent
#         self.agent = AssistantAgent(
#             name="SQL_Researcher",
#             model_client=model_client,
#             tools=[query_tool],
#             system_message="""You are the Data Custodian.
#             Your ONLY job is to find the 9-feature vector and hand it over.
#
#             1. Use 'run_smart_transaction_query' with the CC and Amount provided.
#             2. When you get 'CRITICAL_DATA_VECTOR', repeat it EXACTLY.
#             3. Do not omit numbers. Do not scale numbers.
#             4. If the vector is not 9 items, you have failed your mission."""
#         )