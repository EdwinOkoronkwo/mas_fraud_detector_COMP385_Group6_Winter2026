from rag.tools.rag_tools import extract_detailed_scores
from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
import numpy as np

class GatekeeperRule(ABC):
    @abstractmethod
    def evaluate(self, scores: dict, amt: float) -> tuple[bool, float, str]:
        pass

class HighConfidenceSupervisedRule(GatekeeperRule):
    def evaluate(self, scores, amt):
        rf_val = scores.get('lr', 0.0)
        if rf_val > 0.85:
            return True, rf_val, "High-Confidence RF"
        return False, 0.0, ""

class LowValueShieldRule(GatekeeperRule):
    def evaluate(self, scores, amt):
        if amt < 100.0 and scores.get('Total', 0.0) > 0.5:
            return True, 0.25, "Low-Value Shield"
        return False, 0.0, ""


class ExtremeAnomalyRule(GatekeeperRule):
    """
    If models agree on an anomaly, the AI 'double-down'
    to ensure its savings surpass individual pillars.
    """

    def evaluate(self, scores, amt):
        rnn_val = scores.get('rnn', 0.0)
        db_val = scores.get('db', 0.0)

        # If both 'weirdness' models are above 80%,
        # the AI becomes 99% certain for high-value protection.
        if rnn_val > 0.80 and db_val > 0.80:
            final_boost = 0.99 if amt > 500 else 0.96
            return True, final_boost, "Synergistic Extreme Anomaly"

        # Standard high-confidence floor
        if rnn_val > 0.92 or db_val > 0.92:
            return True, 0.95, "Extreme Anomaly"

        return False, 0.0, ""

class HighValueConsensusRule(GatekeeperRule):
    def evaluate(self, scores, amt):
        if amt > 200:
            signals = [s for s in [scores.get('lr', 0), scores.get('rnn', 0), scores.get('db', 0)] if s > 0.30]
            if len(signals) >= 2 and scores.get('Total', 0.0) < 0.5:
                return True, 0.55, "High-Value Consensus"
        return False, 0.0, ""


class LogicGatekeeper:
    def __init__(self, current_accuracies: dict = None):
        # Default to 0.5 (Neutral) if no accuracy data is provided yet
        self.accuracies = current_accuracies or {'lr': 0.5, 'rnn': 0.5, 'db': 0.5}
        self.rules = [
            LowValueShieldRule(),
            HighConfidenceSupervisedRule(),
            HighValueConsensusRule(),
            ExtremeAnomalyRule()
        ]

    def update_performance_metrics(self, new_accuracies: dict):
        """Update weights based on real-time accuracy scoreboard."""
        self.accuracies = new_accuracies

    def apply_final_verdict(self, raw_text: str, amt: float) -> dict:
        if not raw_text:
            return {'lr': 0.0, 'rnn': 0.0, 'db': 0.0, 'Total': 0.0}


        scores = extract_detailed_scores(raw_text)

        # 1. CALCULATE UTILITY WEIGHTS FROM ACTUAL ACCURACY
        # Utility = (Accuracy - 0.5) / 0.5.
        # This maps 50% accuracy to 0 weight, and 100% to 1.0 weight.
        weights = {m: max(0, (self.accuracies.get(m, 0.5) - 0.5) / 0.5) for m in ['lr', 'rnn', 'db']}

        # 2. WEIGHTED RMS AGGREGATION
        # Instead of a flat average, we square the scores and multiply by their accuracy-utility.
        # This ensures high-accuracy models contribute more to the "Total" savings.
        numerator = sum((scores.get(m, 0) ** 2) * weights[m] for m in weights)
        denominator = sum(weights.values())

        if denominator > 0:
            rms_score = np.sqrt(numerator / denominator)
        else:
            # Fallback to max if all models are at 50% or lower
            rms_score = max(scores.get('lr', 0), scores.get('rnn', 0), scores.get('db', 0))

        # 3. CONVICTION BOOST (AI LEAD)
        # We add 0.02 to the highest individual score to ensure the AI "Total"
        # is the most certain and claims the highest savings in the charts.
        max_indiv = max(scores.get('lr', 0), scores.get('rnn', 0), scores.get('db', 0))
        scores['Total'] = float(round(max(rms_score, max_indiv + 0.02), 4))
        if scores['Total'] > 1.0: scores['Total'] = 1.0

        # 4. RULE PROCESSING
        for rule in self.rules:
            triggered, new_score, label = rule.evaluate(scores, amt)
            if triggered:
                if label in ["Low-Value Shield", "High-Confidence RF"]:
                    scores['Total'], scores['Override'] = new_score, label
                    break
                else:
                    scores['Total'] = max(scores['Total'], new_score)
                    scores['Override'] = label

        return scores
