# FINAL FRAUD DETECTION STRATEGY

## MODEL COMPARISON
| Model Type       | Key Metric               | Value  | Notes                                      |
|------------------|--------------------------|--------|--------------------------------------------|
| Supervised (XGB) | Recall                   | 0.76   | Balanced but misses edge cases.           |
| Neuro (VAE)      | True Positives (TP)      | 166    | Detects anomalies missed by supervised.   |
| Clustering       | Silhouette Score         | 0.2121 | Broad anomaly detection (1297 outliers). |

## CHAMPION SELECTION
**Primary:** `champion_vae.pth` (Neuro)
- **Rationale:** The VAE's 166 TP at 97% precision addresses the critical gap in supervised recall, aligning with the goal of minimizing false negatives in fraud.

**Secondary:** `champion_xgb_dynamic.pkl` (Supervised)
- **Role:** Validates high-confidence predictions and reduces VAE's false positives.

## FEATURES USED (24)
- **Numerical:** `amt`, `zip`, `lat`, `long`, `city_pop`, `unix_time`, `merch_lat`, `merch_long`
- **Categorical:** Transaction categories (e.g., `category_entertainment`), `gender_F`, `gender_M`

## DEPLOYMENT ARCHITECTURE
1. **VAE Layer:** Flags potential fraud for review.
2. **XGB Layer:** Cross-checks VAE outputs to filter noise.
3. **Clustering:** Monitors macro-trends (non-real-time).

## RISK MITIGATION
- **False Positives:** XGB's precision (F1=0.73) filters VAE's outliers.
- **False Negatives:** VAE's 166 TP compensates for XGB's recall gap.

## NEXT STEPS
- A/B test VAE+XGB ensemble vs. XGB alone.
- Monitor clustering drift weekly.