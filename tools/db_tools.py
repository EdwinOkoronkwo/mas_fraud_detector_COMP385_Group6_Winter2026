import pandas as pd
from sqlalchemy import create_engine
import os
from typing import Dict, Any


def migrate_csv_to_sql_tool(kaggle_path: str, db_path: str, sample_frac: float = 0.1) -> str:
    """
    Moves a 10% sample of fraud detection CSV files into a SQLite database.
    """
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)

    files = {
        "train_transactions": os.path.join(kaggle_path, "fraudTrain.csv"),
        "test_transactions": os.path.join(kaggle_path, "fraudTest.csv")
    }

    stats = {}
    try:
        # 1. Ensure the directory exists (Fixes the Windows path issue)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        for table, path in files.items():
            if not os.path.exists(path):
                return f"ERROR: File not found at {path}"

            # 2. Optimization: Only read 10% of the data using the sample_frac
            # We read in chunks to keep memory usage low even during sampling
            total = 0
            for i, chunk in enumerate(pd.read_csv(path, chunksize=100000)):
                # Take 10% of each chunk
                sampled_chunk = chunk.sample(frac=sample_frac, random_state=42)

                mode = 'replace' if i == 0 else 'append'
                sampled_chunk.to_sql(table, engine, if_exists=mode, index=False)
                total += len(sampled_chunk)

            stats[table] = total

        return f"SUCCESS: Migrated 10% sample: {stats['train_transactions']} train rows and {stats['test_transactions']} test rows."
    except Exception as e:
        return f"ERROR: Migration failed: {str(e)}"