# mas_fraud_detector/strategies/supervised/supervised_team.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from tools.training.deployment_tools import persist_champion_model
from tools.training.supervised_common_tools import prepare_championship_data_tool

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.agents import AssistantAgent


def supervised_selector(messages):
    if not messages: return "Supervised_Planner"
    last_source = messages[-1].source

    # Updated Relay Race Sequence
    mapping = {
        "Supervised_Planner": "Sampling_Agent",
        "Sampling_Agent": "Static_XGB_Agent",
        "Static_XGB_Agent": "Dynamic_XGB_Agent",
        "Dynamic_XGB_Agent": "Dynamic_RF_Challenger",  # 🟢 Added RF here
        "Dynamic_RF_Challenger": "ANN_Agent",  # 🟢 RF passes to ANN
        "ANN_Agent": "Supervised_Critic"
    }

    return mapping.get(last_source, "Supervised_Planner")


def create_supervised_team(sampling_agent, static_xgb, dynamic_xgb, rf_agent, ann_agent, critic, model_client, db_path):
    """
    Orchestrates the Supervised Tournament including XGB, RF, and ANN.
    """
    planner = AssistantAgent(
        name="Supervised_Planner",
        model_client=model_client,
        system_message=f"""
        You are the Tournament Director. 
        Your ONLY job is to initiate the sequence: 
        Sampler -> Static XGB -> Dynamic XGB -> Dynamic RF -> ANN -> Critic.

        Do not provide examples of completed tables. 
        Database: {db_path}
        """
    )

    return SelectorGroupChat(
        # Participants must match the order of the relay race
        participants=[planner, sampling_agent, static_xgb, dynamic_xgb, rf_agent, ann_agent, critic],
        model_client=model_client,
        selector_func=supervised_selector,
        termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(50)
    )