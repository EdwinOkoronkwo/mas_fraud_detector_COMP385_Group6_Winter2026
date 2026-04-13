import os
import pandas as pd
import sqlite3

SOURCE_CSV = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\data\kaggle\fraudTest.csv"
# Change this line in your fix_db_types script
BASE_DIR = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
DB_PATH = os.path.join(BASE_DIR, "data", "database.sqlite")

def fix_database_types():
    print("⏳ Loading CSV with strict string types...")
    # Force the CC column to be a string during the read
    df = pd.read_csv(SOURCE_CSV, dtype={'cc_num': str})
    
    conn = sqlite3.connect(DB_PATH)
    print(f"📦 Overwriting {DB_PATH} with precise data...")
    
    # Force the SQL table to use TEXT for cc_num
    df.to_sql("transactions", conn, if_exists="replace", index=False, 
              dtype={'cc_num': 'TEXT'})
    
    # Create an index to make the LIKE search faster
    conn.execute("CREATE INDEX idx_cc_num_text ON transactions(cc_num)")
    
    conn.close()
    print("✅ Done! 16-digit precision is now locked in.")

if __name__ == "__main__":
    fix_database_types()