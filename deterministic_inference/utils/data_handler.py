import sqlite3
import pandas as pd


class DataHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def fetch_balanced_samples(self, n_samples: int) -> pd.DataFrame:
        limit_per_class = n_samples // 2
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        target_table = next((t for t in ["transactions", "test_transactions", "fraud_data"] if t in tables), None)

        if not target_table:
            conn.close()
            raise ValueError(f"Could not find a valid table. Found: {tables}")

        # SURGICAL FIX: Both sides of UNION must use 'actual_label'
        query = f"""
                SELECT * FROM (
                    SELECT CAST(cc_num AS TEXT) as cc_num, 
                           amt, zip, lat, long, city_pop, unix_time, 
                           merch_lat, merch_long, category, gender, 
                           is_fraud as actual_label
                    FROM {target_table} 
                    WHERE is_fraud = 1 LIMIT ?)
                UNION ALL
                SELECT * FROM (
                    SELECT CAST(cc_num AS TEXT) as cc_num, 
                           amt, zip, lat, long, city_pop, unix_time, 
                           merch_lat, merch_long, category, gender, 
                           is_fraud as actual_label
                    FROM {target_table} 
                    WHERE is_fraud = 0 LIMIT ?)
                """

        df = pd.read_sql_query(query, conn, params=[limit_per_class, limit_per_class])
        conn.close()

        # Ensure we return exactly what the pipeline expects
        return df.sample(frac=1).reset_index(drop=True)