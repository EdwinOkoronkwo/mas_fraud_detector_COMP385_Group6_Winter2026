from abc import ABC, abstractmethod
import re


class GatekeeperRule(ABC):
    @abstractmethod
    def evaluate(self, scores: dict, amt: float) -> tuple[bool, float, str]:
        pass


# --- RULES ---
class HighValueFrictionShield(GatekeeperRule):
    def evaluate(self, scores, amt):
        neu_val = scores.get('neural', 0.0)
        clu_val = scores.get('clustering', 0.0)
        if amt > 200.0 and clu_val < 0.1 and neu_val > 0.5:
            return True, 0.12, "High-Value Friction Shield"
        return False, 0.0, ""


class LowValueShieldRule(GatekeeperRule):
    def evaluate(self, scores, amt):
        current_total = scores.get('total', 0.0)

        # If it's a coffee-sized transaction (< $20)
        # and the score is suspicious but not 'Extreme' (< 0.8)
        if amt < 20.0 and current_total < 0.8:
            # We drop it to 0.12 so it falls UNDER your 0.25 threshold
            return True, 0.12, "Micro-Transaction Shield"

        return False, 0.0, ""


class ExtremeAnomalyRule(GatekeeperRule):
    def evaluate(self, scores, amt):
        neu_val = scores.get('neural', 0.0)
        clu_val = scores.get('clustering', 0.0)
        if neu_val > 0.80 and clu_val > 0.80:
            return True, 0.99, "Synergistic Extreme Anomaly"
        return False, 0.0, ""


class MicroTransactionShield(GatekeeperRule):
    """Prevents blocking users for coffee/lunch-sized amounts (< $20)."""

    def evaluate(self, scores, amt):
        current_total = scores.get('total', 0.0)

        # If amount is tiny and it's not a 99% certain 'Extreme Anomaly'
        if amt < 20.0 and current_total < 0.9:
            # Drop it to 0.15 so it falls safely under the 0.25 threshold
            return True, 0.15, "Micro-Transaction Shield"

        return False, 0.0, ""


# --- MAIN CLASS ---
class LogicGatekeeper:
    def __init__(self, registry_weights: dict, threshold: float = 0.25):
        self.weights = registry_weights
        self.threshold = threshold  # Sync with ReportGenerator.THRESHOLD
        self.rules = [
            MicroTransactionShield(), # New Shield for $9 transactions
            HighValueFrictionShield(),
            LowValueShieldRule(),
            ExtremeAnomalyRule()
        ]

    def update_performance_metrics(self, new_accuracies: dict):
        """Re-balances weights based on model performance."""
        raw_votes = {p: max(0.01, new_accuracies.get(p, 0.33)) for p in ['supervised', 'neural', 'clustering']}
        total_points = sum(raw_votes.values())
        for pillar in raw_votes:
            self.weights[pillar] = round(raw_votes[pillar] / total_points, 4)
        return self.weights

    def apply_final_verdict(self, raw_text: str, amt: float) -> dict:
        # 1. Scrape scores (This extracts the 0.0002, 0.3501, 0.2707 AND the 0.6210)
        scores = self._extract_detailed_scores(raw_text)

        # 2. Capture the Agent's "Human-like" Synthesis
        agent_total = scores.get('total', 0.0)

        # 3. Calculate our own fallback math
        weighted_sum = sum(scores[m] * self.weights.get(m, 0.33) for m in ['supervised', 'neural', 'clustering'])

        # 4. DECISION: We trust the Agent's synthesis (0.62) UNLESS
        # the weighted math is somehow even higher (Max-Wins).
        scores['total'] = round(max(agent_total, weighted_sum), 4)
        scores['override'] = "None"

        # 5. Apply "Protective" Rules (Shields)
        for rule in self.rules:
            triggered, new_score, label = rule.evaluate(scores, amt)
            if triggered:
                # Shields ONLY lower the score (to prevent false positives)
                if "Shield" in label:
                    scores['total'] = min(scores['total'], new_score)
                else:
                    scores['total'] = max(scores['total'], new_score)
                scores['override'] = label
                if "Shield" in label: break

        return scores

    def generate_ai_insight(self, scores: dict) -> str:
        total = scores.get('total', 0.0)
        override = scores.get('override', "None")

        # Now reasoning is synced with the verdict
        if total > self.threshold:
            return f"🚨 ALERT: High risk detected ({total:.2f}). Exceeds safety threshold of {self.threshold}."

        if "Shield" in override:
            return f"✅ PROTECTED: {override} applied to prevent customer friction."

        return "⚖️ STABLE: Transaction within normal behavioral bounds."

    def _extract_detailed_scores(self, text: str) -> dict:
        """Internal helper to parse the agent's FINAL_SCORES line."""
        defaults = {'supervised': 0.0, 'neural': 0.0, 'clustering': 0.0, 'total': 0.0}
        pattern = r"LR:\s*([\d\.]+).*?RNN:\s*([\d\.]+).*?DB:\s*([\d\.]+).*?TOTAL:\s*([\d\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                'supervised': float(match.group(1)),
                'neural': float(match.group(2)),
                'clustering': float(match.group(3)),
                'total': float(match.group(4))
            }
        return defaults