import numpy as np


class WeightAdapter:
    def __init__(self, momentum=0.5):
        self.momentum = momentum
        self.agent_performance = {
            "gold": {"tp": 1.0, "fp": 0.01, "fn": 0.01},
            "neuro": {"tp": 0.5, "fp": 0.5, "fn": 0.5},
            "cluster": {"tp": 0.2, "fp": 0.8, "fn": 0.8}
        }
        self.current_weights = {"gold": 0.70, "neuro": 0.20, "cluster": 0.10}

    def get_weights(self):
        """🚀 FIX: The method the BatchProcessor was looking for."""
        return self.current_weights

    def update_performance(self, actual, gold_p, n_p, c_p, threshold=0.3):
        """Updates stats based on Ground Truth."""
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
        scores = {}
        for name, stats in self.agent_performance.items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"])
            recall = stats["tp"] / (stats["tp"] + stats["fn"])
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.01
            scores[name] = f1 ** 2

        total = sum(scores.values())
        if total == 0: return  # Safety check

        new_weights = {k: (v / total) for k, v in scores.items()}

        for k in self.current_weights:
            self.current_weights[k] = (1 - self.momentum) * self.current_weights[k] + (self.momentum * new_weights[k])