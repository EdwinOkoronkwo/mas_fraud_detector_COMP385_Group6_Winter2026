from utils.visualizer import VisualizerFacade
import os  # 🟢 ADD THIS LINE
import sqlite3


def execute_comprehensive_eda(db_path: str, output_dir: str) -> str:
    import sqlite3
    import pandas as pd
    import os

    # 🟢 Ensure the directory exists physically
    os.makedirs(output_dir, exist_ok=True)
    abs_output_path = os.path.abspath(output_dir)

    # 1. Connect and Load Raw Data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM train_transactions", conn)
    conn.close()

    # 2. Initialize the Facade
    facade = VisualizerFacade(output_dir=abs_output_path)

    # 3. Structural Analysis
    markdown_report = facade.inspector.get_summary_table(df, "Raw Data Exploration")

    # 4. Statistical Analysis (Note: Ensure your plotters use the facade's internal path)
    # If your plotters expect (dataframe, save_path), use os.path.join:
    facade.dist_plotter.plot_class_balance(df, os.path.join(abs_output_path, "class_balance.png"))
    facade.corr_plotter.plot_heatmap(df, os.path.join(abs_output_path, "correlation_heatmap.png"))
    facade.cat_plotter.plot_fraud_by_category(df, os.path.join(abs_output_path, "fraud_by_category.png"))
    facade.dist_plotter.plot_financial_forensics(df, os.path.join(abs_output_path, "financial_forensics.png"))

    # Log to terminal for your peace of mind
    print(f"✅ EDA Plots successfully written to: {abs_output_path}")

    return markdown_report