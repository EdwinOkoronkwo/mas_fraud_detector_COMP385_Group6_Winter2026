import os

import joblib
from xgboost import XGBClassifier


def recreate_baseline_champion(self):
    """Recreates the baseline using the high-standard Champion logic."""
    self.logger.info("🔄 RECREATING BASELINE: Aligning with Champion Standards...")

    # 1. Load the same 27-feature bundle
    data_bundle = joblib.load(self.settings.TEMP_SPLIT_PATH)
    X_train, y_train = data_bundle['train']

    # 2. Use the "Champion" Parameter Space
    # We use the parameters that gave the Champion 0.71 F1 in the tournament
    baseline_champion = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=12,  # To handle the imbalance SMOTE might have missed
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )

    # 3. Fit on the exact same scaled data
    baseline_champion.fit(X_train, y_train)

    # 4. Save to the proper directory
    save_path = os.path.join(self.settings.MODELS_DIR, "baselines", "manual_xgb_baseline.pkl")
    joblib.dump(baseline_champion, save_path)

    self.logger.info(f"✅ Baseline Recreated. Path: {save_path}")