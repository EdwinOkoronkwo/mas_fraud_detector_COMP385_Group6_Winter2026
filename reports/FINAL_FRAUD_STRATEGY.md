# FINAL FRAUD DETECTION STRATEGY

## MODEL COMPARISON
| Model Type       | Key Metric               | Value  | Notes                                      |
|------------------|--------------------------|--------|--------------------------------------------|
| Supervised (XGB) | Recall                   | 0.76   | Balanced but missed ~24% of fraud cases   |
| Neuro (VAE)      | True Positives (Anomaly) | 166    | Captured high-risk cases missed by XGB    |
| Clustering       | Silhouette Score         | 0.2121 | Broad anomaly detection (1297 flagged)    |

## CHAMPION SELECTION
**Primary:** `champion_vae.pth` (Neuro)
- **Rationale:** The VAE's 166 true positives at 97% precision address the XGB's recall gap, making it ideal for high-stakes fraud detection where false negatives are costly.

**Secondary:** `champion_xgb_dynamic.pkl` (Supervised)
- **Rationale:** Used for explainability and baseline validation. Recall of 0.76 ensures broad coverage.

## FEATURES USED (27)
- **Numerical:** `amt`, `zip`, `lat`, `long`, `city_pop`, `unix_time`, `merch_lat`, `merch_long`, `amt_to_cat_avg`, `high_risk_time`, `txn_velocity`
- **Categorical:** One-hot encoded transaction categories (`entertainment`, `food_dining`, etc.) and `gender`.

## DEPLOYMENT ARCHITECTURE
1. **Real-Time Pipeline:** VAE flags anomalies → XGB validates high-probability cases.
2. **Fallback:** K-Means clusters for macro-level monitoring.

## RISK MITIGATION
- **VAE Threshold Tuning:** Adjust `threshold_p` to balance precision/recall.
- **Feature Refresh:** Quarterly review of `amt_to_cat_avg` and `txn_velocity`.

---