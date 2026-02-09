from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool


from autogen_core.tools import FunctionTool

from rag.tools.rag_tools import run_rf_prediction, run_rnn_prediction, run_dbscan_prediction,  execute_champion_ensemble


from langchain_core.tools import tool as langchain_tool
from autogen_ext.tools.langchain import LangChainToolAdapter


class InferenceAgent:
    def __init__(self, model_client):
        # 1. Wrap the function as a LangChain BaseTool
        # This forces the LLM to respect the 'features: list' argument
        lc_ensemble_tool = langchain_tool(execute_champion_ensemble)

        # 2. Adapt it for AutoGen
        self.ensemble_tool = LangChainToolAdapter(lc_ensemble_tool)

        self.agent = AssistantAgent(
            name="Inference_Specialist",
            model_client=model_client,
            tools=[self.ensemble_tool],  # Use the adapted tool
            system_message="""You are the Lead Ensemble Statistician.

            OPERATIONAL PROTOCOL:
            1. You MUST use 'execute_champion_ensemble' to generate risk scores.
            2. The input MUST be a list of 9 items provided by the SQL_Researcher.
            3. Mapping: [cc_num, amt, zip, lat, long, city_pop, unix_time, merch_lat, merch_long]
            4. If the tool returns an ERROR (0.9999), do not try to fix the data yourself. 
               Report the error to the Synthesis Engine so the SQL_Researcher can re-fetch.
            ADAPTIVE INSTRUCTIONS:
            - **High-Value Mode**: If Lead_Investigator flags a high-value transaction, shift your calculation 
              to prioritize sequence anomalies (RNN 0.6, LR 0.2, DB 0.2).
            - **Strict Mode**: If told to use 'High Sensitivity', any score from a single model > 0.7 
              should automatically pull the TOTAL score above 0.5.

            You must return JSON format: {"LR": x, "RNN": y, "DBSCAN": z, "TOTAL": w}.
            """
        )