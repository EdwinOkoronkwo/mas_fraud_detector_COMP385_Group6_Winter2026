import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Gold Model and Feature List
model = joblib.load('models/gold_xgb.pkl')
# Assuming your features are stored in the model or a separate list
features = model.feature_names_in_

# 2. Extract Importances
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# 3. Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top Decision Drivers: Gold Pillar (27 Features)')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('reports/feature_importance.png')
plt.show()

print("✅ Feature Importance plot saved to reports/feature_importance.png")