import sqlite3
import pandas as pd

# Path to your verified database
db_path = r'C:\Youtube\autogen\mas_fraud_detector\data\database.sqlite'

import sqlite3
import pandas as pd

db_path = r"C:\Youtube\autogen\mas_fraud_detector\data\database.sqlite"
conn = sqlite3.connect(db_path)

# Check the UN-SCALED table
df = pd.read_sql_query("SELECT * FROM train_transactions LIMIT 1", conn)
print("\n--- COLUMNS IN TRAIN_TRANSACTIONS ---")
print(df.columns.tolist())
print("\n--- SAMPLE ROW ---")
print(df.iloc[0])
conn.close()
def run_inspection():
    try:
        conn = sqlite3.connect(db_path)

        # 1. Get all table names
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print("=== TABLES FOUND ===")
        print(tables)

        for table in tables['name']:
            print(f"\n--- Analysis for table: {table} ---")

            # 2. Get Row Count
            count = pd.read_sql_query(f"SELECT COUNT(*) as total FROM {table}", conn)
            print(f"Total Rows: {count['total'][0]:,}")

            # 3. Check Fraud Balance (if the column exists)
            try:
                balance = pd.read_sql_query(f"SELECT is_fraud, COUNT(*) as count FROM {table} GROUP BY is_fraud", conn)
                print("Class Distribution:")
                print(balance)
            except:
                print("Column 'is_fraud' not found in this table.")


        conn.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")


if __name__ == "__main__":
    run_inspection()