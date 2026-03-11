import json
import sqlite3

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool


class SQLResearcher:
    def __init__(self, model_client, db_path):
        self.db_path = db_path

        def run_smart_transaction_query(cc_num: str, amt: float) -> str:
            try:
                conn = sqlite3.connect(self.db_path)

                # FIX 1: Use LIKE for CC Tail (e.g., %9027)
                # FIX 2: Use BETWEEN for float precision (amt +/- 0.01)
                query = """
                        SELECT * \
                        FROM test_transactions
                        WHERE cc_num LIKE ?
                          AND amt BETWEEN ? AND ? LIMIT 1 \
                        """

                # We add the wildcard % to handle the tail match
                cc_search = f"%{cc_num}"
                df = pd.read_sql_query(query, conn, params=[cc_search, amt - 0.01, amt + 0.01])
                conn.close()

                if df.empty:
                    return "[ERROR]: No record found. Advise Vector_Researcher to use Math context only."

                data_dict = df.iloc[0].to_dict()
                data_dict['cc_num'] = str(data_dict['cc_num'])

                return f"CRITICAL_DATA_JSON: {json.dumps(data_dict)}"
            except Exception as e:
                return f"[ERROR]: {str(e)}"

        query_tool = FunctionTool(
            run_smart_transaction_query,
            name="run_smart_transaction_query",
            description="Fetches full transaction record using CC tail and amount range matching."
        )

        self.agent = AssistantAgent(
            name="SQL_Researcher",
            model_client=model_client,
            tools=[query_tool],
            system_message="""You are the Data Custodian. 
            1. Use 'run_smart_transaction_query' to fetch the record.
            2. If you get a record, provide the CRITICAL_DATA_JSON.
            3. If no record is found, state 'DATA_MISSING' so the next agent knows to rely on Math scoring."""
        )