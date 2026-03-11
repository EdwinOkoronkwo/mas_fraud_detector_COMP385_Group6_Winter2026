import numpy as np

import numpy as np

import numpy as np

from deterministic_inference.core.weight_adapter import WeightAdapter


class ScoringEngine:
    def __init__(self, adapter=None):
        self.adapter = adapter or WeightAdapter()
        # Default Trinity Ratios (Balanced)
        self.default_weights = {"gold": 0.4, "neuro": 0.4, "cluster": 0.2}

    def _normalize_neuro(self, mse_loss):
        # 🚀 CALIBRATION: Centered at 0.12 because raw MSE is currently < 0.2
        # Slope 0.04 provides a sharp but smooth ramp-up
        return 1 / (1 + np.exp(-(mse_loss - 0.12) / 0.04))

    def _normalize_cluster(self, raw_dist):
        # 🚀 CALIBRATION: Centered at 2.2 based on your CLU_RAW audit trail
        # This allows raw distances > 3.0 to hit near-perfect anomaly scores
        return 1 / (1 + np.exp(-(raw_dist - 2.2) / 0.3))

    def compute_mas_score(self, gold_prob, neuro_mse, cluster_dist):
        n_p = self._normalize_neuro(neuro_mse)
        c_p = self._normalize_cluster(cluster_dist)

        # Get the latest optimized weights from the adapter
        dynamic_base = self.adapter.get_weights()

        # --- REFINED DYNAMIC WEIGHTING ---
        if gold_prob < 0.2 and n_p > 0.7 and c_p > 0.6:
            # 🕵️ STEALTH SENSOR: Gold is blind, but Anomaly & Cluster are screaming
            weights = {"gold": 0.1, "neuro": 0.6, "cluster": 0.3}
            mode = "STEALTH_OVERRIDE"
        elif gold_prob < 0.05:
            # 🛡️ VETO: If Gold is certain it's safe, suppress weak anomaly signals
            weights = {"gold": 0.9, "neuro": 0.05, "cluster": 0.05}
            mode = "VETO_PROTECT"
        else:
            # 📈 ADAPTIVE: Standard weighted blend
            weights = dynamic_base
            mode = "PERFORMANCE_ADAPTIVE"

        final_score = (gold_prob * weights["gold"]) + \
                      (n_p * weights["neuro"]) + \
                      (c_p * weights["cluster"])

        return {
            "final_score": float(final_score),
            "n_p": float(n_p),
            "c_p": float(c_p),
            "mode": mode,
            "weights": weights
        }