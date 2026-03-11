import sqlite3
import pandas as pd
from config.settings import DB_PATH


def audit_database():
    print(f"🔍 AUDITING DATABASE: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"📂 Found Tables: {tables}")

    for table in tables:
        print(f"\n--- Table: {table} ---")

        # 2. Get Column Names and Types
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        print(f"📊 Columns: {[c[1] for c in columns]}")

        # 3. Show first 3 rows to check values (Scaled vs Raw)
        try:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
            print("📝 Sample Data:")
            print(df)
        except Exception as e:
            print(f"❌ Could not read data: {e}")

    conn.close()


if __name__ == "__main__":
    audit_database()