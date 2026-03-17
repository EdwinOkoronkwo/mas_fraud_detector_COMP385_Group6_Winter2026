import numpy as np

class WeightAdapter:
    def __init__(self, momentum=0.4):
        self.momentum = momentum
        self.agent_performance = {
            "gold": {"tp": 0.5, "fp": 0, "fn": 0.5},
            "neuro": {"tp": 0.5, "fp": 0, "fn": 0.5},
            "cluster": {"tp": 0.5, "fp": 0, "fn": 0.5}
        }
        self.current_weights = {"gold": 0.34, "neuro": 0.33, "cluster": 0.33}

    # 🚀 THIS IS THE MISSING PIECE
    def get_weights(self):
        """Returns the current trust weights for the agents."""
        return self.current_weights

    def update_performance(self, actual, gold_p, n_p, c_p, threshold=0.3):
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
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.5
            # Recall with the 2x Penalty for False Negatives to beat the baseline
            recall = stats["tp"] / (stats["tp"] + (stats["fn"] * 2.0))
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.1
            scores[name] = f1

        total = sum(scores.values())
        new_weights = {k: (v / total) for k, v in scores.items()}

        for k in self.current_weights:
            self.current_weights[k] = (1 - self.momentum) * self.current_weights[k] + (self.momentum * new_weights[k])