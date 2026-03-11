# mas_fraud_detector/strategies/supervised/supervised_team.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from mas_fraud_detector.tools.training.deployment_tools import persist_champion_model
from mas_fraud_detector.tools.training.supervised_common_tools import prepare_championship_data_tool

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.agents import AssistantAgent


# def supervised_selector(messages):
#     msg_history = str(messages).lower()
#     last_msg = str(messages[-1].content).lower() if messages else ""
#
#     # Fallback to Critic to ensure termination if history gets messy
#     selected_agent = "Model_Critic"
#
#     # STATE 1: Data Preparation
#     if "success: training set balanced" not in msg_history:
#         selected_agent = "Sampling_Agent"
#
#     # STATE 2: Sequential Model Training
#     elif "success: training set balanced" in last_msg:
#         selected_agent = "LR_Agent"
#
#     elif '"model": "logistic regression"' in last_msg or "logistic regression error" in last_msg:
#         selected_agent = "RF_Agent"
#
#     elif '"model": "random forest"' in last_msg or "random forest error" in last_msg:
#         selected_agent = "XGB_Agent"
#
#     elif '"model": "extreme gradient boosting"' in last_msg or "extreme gradient boosting error" in last_msg:
#         selected_agent = "ANN_Agent"
#
#     # STATE 3: Persistence & Audit Handshake
#     # If ANN is done, OR if the persistence tool just finished, give it to the Critic
#     elif '"model": "artificial neural network"' in last_msg or "artificial neural network error" in last_msg:
#         selected_agent = "Model_Critic"
#
#     print(f"--- [SELECTOR]: Choosing {selected_agent} based on last message keywords ---")
#     return selected_agent
def supervised_selector(messages):
    msg_history = str(messages).lower()
    last_msg = str(messages[-1].content).lower() if messages else ""

    # Default to Planner if we are just starting
    if len(messages) <= 1:
        return "Supervised_Planner"

    # STATE 1: Data Preparation
    if "success: training set balanced" not in msg_history:
        return "Sampling_Agent"

    # STATE 2: Sequential Model Training (Expert Decisions)
    # Transition from Sampling to the first Model (LR)
    if "success: training set balanced" in last_msg:
        return "LR_Agent"

    # Transition from LR to RF
    if '"model": "logistic regression"' in last_msg or "logistic regression error" in last_msg:
        return "RF_Agent"

    # Transition from RF to XGB
    if '"model": "random forest"' in last_msg or "random forest error" in last_msg:
        return "XGB_Agent"

    # Transition from XGB to ANN
    if '"model": "extreme gradient boosting"' in last_msg or "extreme gradient boosting error" in last_msg:
        return "ANN_Agent"

    # STATE 3: Final Audit
    if '"model": "artificial neural network"' in last_msg or "artificial neural network error" in last_msg:
        return "Model_Critic"

    return "Model_Critic"

def create_supervised_team(sampling_agent, lr_agent, rf_agent, xgb_agent, ann_agent, model_client, db_path):
    # Tool wrapper for the Critic
    def persist_champion(model_name: str, metrics_json: str) -> str:
        return persist_champion_model(model_name, metrics_json)

    planner = AssistantAgent(
        name="Supervised_Planner",
        model_client=model_client,
        system_message="""DIRECTOR ROLE: 
        1. Coordinate Sampling_Agent -> Models -> Model_Critic.
        RESTRICTION: Zero-talk. Just direct traffic."""
    )
    critic = AssistantAgent(
        name="Model_Critic",
        model_client=model_client,
        system_message="""AUDITOR ROLE: 
        1. Compare the results of the LR, RF, XGB, and ANN specialists.
        2. Review the 'parameters_used' section in their reports.
        3. If a model has a high Recall but very low F1 (too many false positives), 
           critique the Agent's choice of hyperparameters.
        4. Format a Markdown table of results.
        5. DECLARE the CHAMPION based on the best balance of Fraud Detection vs. Operational Noise.
        6. End with 'TERMINATE'."""
    )

    # critic = AssistantAgent(
    #     name="Model_Critic",
    #     model_client=model_client,
    #     system_message="""AUDITOR ROLE: Final Report Mode.
    #         1. Compare metrics for LR, RF, XGB, and ANN.
    #         2. Format a Markdown table.
    #         3. Clearly state: 'CHAMPION_MODEL: [Model Name]'
    #         4. End with 'TERMINATE'."""
    # )

    return SelectorGroupChat(
        participants=[planner, sampling_agent, lr_agent, rf_agent, xgb_agent, ann_agent, critic],
        model_client=model_client,
        selector_func=supervised_selector,
        termination_condition=MaxMessageTermination(max_messages=25) | TextMentionTermination("TERMINATE")
    )