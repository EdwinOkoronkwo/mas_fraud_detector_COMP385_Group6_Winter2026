import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Paths
DB_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\data\database.sqlite"
OUTPUT_DIR = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\reports\eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_plots():
    print(f"Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Load the raw data for EDA
    df = pd.read_sql("SELECT * FROM train_transactions", conn)
    conn.close()

    # --- PLOT 1: Class Imbalance ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_fraud', data=df, palette='viridis')
    plt.title('Figure 3.1: Transaction Class Distribution (1=Fraud)')
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    print("✅ Generated: class_distribution.png")

    # --- PLOT 2: Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    # Selecting only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Figure 3.2: Feature Correlation Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    print("✅ Generated: correlation_heatmap.png")

    # --- PLOT 3: Fraud by Category ---
    plt.figure(figsize=(12, 6))
    cat_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    sns.barplot(x=cat_fraud.index, y=cat_fraud.values, palette='Reds_r')
    plt.xticks(rotation=45)
    plt.title('Figure 3.3: Probability of Fraud by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fraud_by_category.png"))
    print("✅ Generated: fraud_by_category.png")


if __name__ == "__main__":
    generate_plots()
    print(f"\n🚀 SUCCESS! View your plots here: {OUTPUT_DIR}")