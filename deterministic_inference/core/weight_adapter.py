import numpy as np


class WeightAdapter:
    def __init__(self, momentum=0.3):  # Increased momentum for more visible curve
        self.momentum = momentum
        self.agent_performance = {
            # Low seeds (0.1) make the system react faster to early mistakes
            "gold": {"tp": 0.1, "fp": 0, "fn": 0.1},
            "neuro": {"tp": 0.1, "fp": 0, "fn": 0.1},
            "cluster": {"tp": 0.1, "fp": 0, "fn": 0.1}
        }
        # Start everyone equal so we can watch them diverge
        self.current_weights = {"gold": 0.33, "neuro": 0.33, "cluster": 0.34}

    def update_performance(self, actual, gold_p, n_p, c_p, threshold=0.3):
        """Updates internal stats based on whether agents were right or wrong."""
        agents = {"gold": gold_p, "neuro": n_p, "cluster": c_p}

        for name, prob in agents.items():
            pred = 1 if prob >= threshold else 0
            if pred == 1 and actual == 1:
                self.agent_performance[name]["tp"] += 1
            elif pred == 1 and actual == 0:
                self.agent_performance[name]["fp"] += 1
            elif pred == 0 and actual == 1:
                self.agent_performance[name]["fn"] += 1

        self._recalculate_weights()

    def _recalculate_weights(self):
        """Calculates new weights based on F1-Score of each agent."""
        scores = {}
        for name, stats in self.agent_performance.items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"])
            recall = stats["tp"] / (stats["tp"] + stats["fn"])
            # Calculate F1 as the trust metric
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.1
            scores[name] = f1

        # Normalize scores so they sum to 1.0
        total = sum(scores.values())
        new_weights = {k: (v / total) for k, v in scores.items()}

        # Apply momentum (Smooth transition so weights don't jitter)
        for k in self.current_weights:
            self.current_weights[k] = (1 - self.momentum) * self.current_weights[k] + (self.momentum * new_weights[k])

    def get_weights(self):
        return self.current_weights