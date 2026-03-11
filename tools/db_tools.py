import pandas as pd
from sqlalchemy import create_engine
import os
from typing import Dict, Any


def migrate_csv_to_sql_tool(kaggle_path: str, db_path: str, sample_frac: float = 0.1) -> str:
    """
    Moves both Train and Test CSV files into a SQLite database.
    """
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        results = []

        # List of files to look for and their corresponding table names
        data_map = {
            "fraudTrain.csv": "train_transactions",
            "fraudTest.csv": "test_transactions"
        }

        for csv_name, table_name in data_map.items():
            file_path = os.path.join(kaggle_path, csv_name)

            if os.path.exists(file_path):
                # Read, sample, and migrate
                df = pd.read_csv(file_path)

                # Use a larger sample for testing to ensure we have enough fraud cases
                current_sample = df.sample(frac=sample_frac, random_state=42)
                current_sample.to_sql(table_name, engine, if_exists='replace', index=False)
                results.append(f"{table_name} ({len(current_sample)} rows)")
            else:
                results.append(f"SKIPPED: {csv_name} not found.")

        return f"SUCCESS: Migrated {', '.join(results)}. INGESTION_COMPLETE"
    except Exception as e:
        return f"ERROR: {str(e)}"

def purge_existing_database(db_path: str):
    """
    Forcefully removes all tables and shrinks the database file.
    """
    import sqlite3
    import os

    # Resolve absolute path to avoid "wrong file" issues
    abs_path = os.path.abspath(db_path)

    try:
        # Use a context manager to ensure the connection CLOSES
        with sqlite3.connect(abs_path) as conn:
            cursor = conn.cursor()

            # Disable foreign key checks to allow dropping everything
            cursor.execute("PRAGMA foreign_keys = OFF;")

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cursor.fetchall() if not t[0].startswith("sqlite_")]

            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table};")
                print(f"🗑️ Dropped table: {table}")

            # VACUUM cleans the database file physically
            cursor.execute("VACUUM;")
            conn.commit()

        return f"SUCCESS: Database at {abs_path} purged and vacuumed."
    except Exception as e:
        return f"ERROR during purge: {str(e)}"