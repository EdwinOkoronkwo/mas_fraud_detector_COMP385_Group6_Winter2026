import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataInspector:
    """Provides the 'thorough' tabular data exploration."""

    def get_summary_table(self, df: pd.DataFrame, stage_name: str) -> str:
        report = f"## EDA Report: {stage_name}\n"
        report += f"### Dataset Head (Top 5 Rows)\n{df.head().to_markdown()}\n\n"
        report += f"### Feature Schema & Data Types\n"

        # Capture df.info() style data into a table
        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count().values,
            'Dtype': df.dtypes.values
        })
        report += schema_df.to_markdown(index=False) + "\n\n"
        return report


class DistributionPlotter:
    """Handles the visual 'Before and After' comparisons."""
    def plot_class_balance(self, df: pd.DataFrame, path: str, suffix: str):
        plt.figure(figsize=(7, 5))
        sns.countplot(x='is_fraud', data=df, palette='magma')
        plt.title(f"Class Distribution - {suffix}")
        plt.savefig(f"{path}/class_dist_{suffix.lower()}.png")
        plt.close()

    def plot_financial_forensics(self, df: pd.DataFrame, path: str):
        """Standard EDA plot for fraud: Amount vs. Fraud status."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='amt', hue='is_fraud', kde=True, element="step")
        plt.xlim(0, 500) # Focusing on the bulk of transactions
        plt.title("Transaction Amount Distribution by Class")
        plt.savefig(f"{path}/amt_hist.png")
        plt.close()


class CorrelationPlotter:
    """Responsible for relationship mapping."""

    def plot_heatmap(self, df: pd.DataFrame, path: str):
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=['number']).corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt=".2f")
        plt.title("Pearson Correlation Matrix")
        plt.savefig(f"{path}/correlation_matrix.png")
        plt.close()

class CategoricalPlotter:
    """Analyzes fraud frequency across categorical dimensions."""
    def plot_fraud_by_category(self, df: pd.DataFrame, path: str):
        plt.figure(figsize=(12, 6))
        # Calculate fraud percentage per category
        cat_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
        sns.barplot(x=cat_fraud.index, y=cat_fraud.values, palette='Reds_r')
        plt.xticks(rotation=45)
        plt.title("Fraud Probability by Transaction Category")
        plt.ylabel("Mean Fraud Rate")
        plt.savefig(f"{path}/category_fraud_rate.png")
        plt.close()


class VisualizerFacade:
    def __init__(self, output_dir: str = "reports/eda"):
        # 🟢 FORCE ABSOLUTE PATH
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Log this so you can see exactly where it's going in your console
        print(f"DEBUG: Visualizer initialized. Saving plots to: {self.output_dir}")

        self.inspector = DataInspector()
        self.dist_plotter = DistributionPlotter()
        self.corr_plotter = CorrelationPlotter()
        self.cat_plotter = CategoricalPlotter()
    def analyze_raw_data(self, df: pd.DataFrame) -> str:
        """Full EDA for the initial state."""
        report = self.inspector.get_summary_table(df, "Initial Ingestion")

        # Trigger all composed plotting engines
        self.dist_plotter.plot_class_balance(df, self.output_dir, "Pre-SMOTE")
        self.dist_plotter.plot_financial_forensics(df, self.output_dir)
        self.corr_plotter.plot_heatmap(df, self.output_dir)
        self.cat_plotter.plot_fraud_by_category(df, self.output_dir)  # Added this

        return report

    def verify_resampling(self, df: pd.DataFrame):
        """Specific visual check for post-SMOTE state."""
        self.dist_plotter.plot_class_balance(df, self.output_dir, "Post-SMOTE")