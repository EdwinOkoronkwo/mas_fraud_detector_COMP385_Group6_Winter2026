import pytest
from autogen_agentchat.messages import TextMessage

from strategies.supervised.supervised_team import supervised_selector

"""
MODULE: Supervised Tournament Integration Testing
PURPOSE: 
This suite verifies the 'Relay Race' orchestration of the Model Tournament.
Unlike the Preprocessing pipeline which relies on keyword 'Signals', this 
selector uses 'Source-Based Switching' to move the baton between experts.

SEQUENCE: 
Planner -> Sampler -> Static XGB -> Dynamic XGB -> Dynamic RF -> ANN -> Critic.
"""

class TestSupervisedIntegration:

    def test_tournament_initialization(self):
        """
        PURPOSE: Verify the Tournament Director starts the race.
        LOGIC: If the conversation history is empty, the selector must
        always default to the 'Supervised_Planner'.
        """
        assert supervised_selector([]) == "Supervised_Planner"

    def test_relay_race_handover(self):
        """
        PURPOSE: Verify the full specialist sequence from XGB to ANN.
        LOGIC: Each agent's name as a 'source' must trigger the next
        specific specialist in the mapping.
        """
        # 1. Planner starts the Sampler
        msgs = [TextMessage(content="Begin tournament", source="Supervised_Planner")]
        assert supervised_selector(msgs) == "Sampling_Agent"

        # 2. Sampler passes to Static Baseline
        msgs.append(TextMessage(content="Data sampled.", source="Sampling_Agent"))
        assert supervised_selector(msgs) == "Static_XGB_Agent"

        # 3. Static Baseline passes to Autonomous Optimizer
        msgs.append(TextMessage(content="Baseline: F1=0.70", source="Static_XGB_Agent"))
        assert supervised_selector(msgs) == "Dynamic_XGB_Agent"

    def test_new_challenger_sequence(self):
        """
        PURPOSE: Verify the integration of the RF and ANN (TensorFlow) specialists.
        LOGIC: Confirms that the Dynamic XGB correctly hands off to the
        Random Forest, and the RF hands off to the ANN.
        """
        # 1. XGB -> RF Handover
        msgs = [TextMessage(content="XGB optimized: F1=0.76", source="Dynamic_XGB_Agent")]
        assert supervised_selector(msgs) == "Dynamic_RF_Challenger"

        # 2. RF -> ANN Handover (The Deep Learning transition)
        msgs.append(TextMessage(content="RF results ready.", source="Dynamic_RF_Challenger"))
        assert supervised_selector(msgs) == "ANN_Agent"

        # 3. ANN -> Critic (The Final Audit)
        msgs.append(TextMessage(content="ANN (TF) training complete.", source="ANN_Agent"))
        assert supervised_selector(msgs) == "Supervised_Critic"

    def test_selector_default_behavior(self):
        """
        PURPOSE: Prevent infinite loops or dead states.
        LOGIC: If an unknown source speaks, the system should reset to
        the Planner rather than crashing or halting.
        """
        msgs = [TextMessage(content="Hello?", source="Unknown_Agent")]
        assert supervised_selector(msgs) == "Supervised_Planner"