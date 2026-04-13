import json
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

    def get_transaction_by_cc(self, cc_num: str) -> pd.DataFrame:
        # 1. Connect to the SAME db_path used by fetch_balanced_samples
        conn = sqlite3.connect(self.db_path)
        
        # 2. We are looking for the 'transactions' table specifically
        # (This is the table created by your fix_db_types script)
        target_table = "transactions" 

        # 3. Use an EXACT string match to avoid rounding errors
        query = f"""
                SELECT 
                    CAST(cc_num AS TEXT) as cc_num, 
                    merchant, category, amt, first, last, gender, 
                    street, city, state, zip, lat, long, city_pop, 
                    job, dob, trans_num, unix_time, merch_lat, merch_long, 
                    is_fraud as actual_label
                FROM {target_table}
                WHERE cc_num = ? 
                LIMIT 1
            """

        # Ensure the CC is a clean string
        search_val = str(cc_num).strip()
        
        try:
            df = pd.read_sql_query(query, conn, params=[search_val])
        except Exception as e:
            print(f"Error querying {target_table}: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()

        # 4. Cleanup the merchant names for the UI
        if not df.empty:
            df['merchant'] = df['merchant'].str.replace(r'^fraud_', '', regex=True)
            df['merchant'] = df['merchant'].str.replace('_', ' ').str.title().str.strip()
            df['actual_label'] = df['actual_label'].astype(int)

        return df
    # def get_transaction_by_cc(self, cc_num: str) -> pd.DataFrame:
    #     conn = sqlite3.connect(self.db_path)
    #     cursor = conn.cursor()

    #     # 1. Table Discovery Logic
    #     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #     tables = [t[0] for t in cursor.fetchall()]
    #     target_table = next((t for t in ["transactions", "test_transactions", "fraud_data"] if t in tables), None)

    #     if not target_table:
    #         conn.close()
    #         return pd.DataFrame()

    #     # 2. Comprehensive Query: All columns required for Model Preprocessing + RAG Audit
    #     # 2. Comprehensive Query
    #     # Remove CAST from the WHERE clause to prevent SQLite numeric rounding
    #     query = f"""
    #             SELECT 
    #                 CAST(cc_num AS TEXT) as cc_num, 
    #                 merchant, category, amt, first, last, gender, 
    #                 street, city, state, zip, lat, long, city_pop, 
    #                 job, dob, trans_num, unix_time, merch_lat, merch_long, 
    #                 is_fraud as actual_label
    #             FROM {target_table}
    #             WHERE cc_num LIKE ? 
    #             LIMIT 1
    #         """

    #     # Use a wildcard to ensure we find the string representation accurately
    #     df = pd.read_sql_query(query, conn, params=[f"%{cc_num}%"])
    #     conn.close()

    #     # 3. Professional Data Cleanup
    #     if not df.empty:
    #         # Remove "fraud_" prefix
    #         df['merchant'] = df['merchant'].str.replace(r'^fraud_', '', regex=True)

    #         # 🚀 EXTRA POLISH: Replace underscores with spaces and Title Case the names
    #         # Transforms 'romaguera_wehner' -> 'Romaguera Wehner'
    #         df['merchant'] = df['merchant'].str.replace('_', ' ').str.title().str.strip()

    #         # Ensure 'actual_label' is integer type for the math models
    #         df['actual_label'] = df['actual_label'].astype(int)

    #     return df


    def load_audit_history(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves the history of all AI-generated fraud audits."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM audit_history ORDER BY rowid DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=[limit])

            # Deserialize the trace back into Python objects for the UI
            if 'trace' in df.columns:
                df['trace'] = df['trace'].apply(lambda x: json.loads(x) if x else [])
            return df
        except:
            return pd.DataFrame()  # Returns empty if table doesn't exist yet
        finally:
            conn.close()