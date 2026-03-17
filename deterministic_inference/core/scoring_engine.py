import numpy as np


from deterministic_inference.core.weight_adapter import WeightAdapter




class ScoringEngine:
    def __init__(self, adapter=None):
        self.adapter = adapter or WeightAdapter()
        
    def _normalize_neuro(self, mse_loss):
        return 1 / (1 + np.exp(-(mse_loss - 0.18) / 0.03))

    def _normalize_cluster(self, raw_dist):
        return 1 / (1 + np.exp(-(raw_dist - 4.0) / 0.6))

    def compute_mas_score(self, gold_prob, neuro_mse, cluster_dist):
        n_p = self._normalize_neuro(neuro_mse)
        c_p = self._normalize_cluster(cluster_dist)
        
        # This will now work correctly
        dynamic_base = self.adapter.get_weights()

        if gold_prob < 0.02:
            weights = {"gold": 0.90, "neuro": 0.05, "cluster": 0.05}
            mode = "VETO_STRICT"
        elif n_p > 0.75 or c_p > 0.80:
            weights = {"gold": 0.15, "neuro": 0.45, "cluster": 0.40}
            mode = "SENSOR_OVERRIDE"
        else:
            weights = dynamic_base
            mode = "PERFORMANCE_ADAPTIVE"

        final_score = (gold_prob * weights["gold"]) + (n_p * weights["neuro"]) + (c_p * weights["cluster"])

        return {
            "final_score": float(final_score),
            "n_p": float(n_p),
            "c_p": float(c_p),
            "mode": mode,
            "weights": weights
        }