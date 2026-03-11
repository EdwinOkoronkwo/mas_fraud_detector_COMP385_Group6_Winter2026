import sqlite3
import pandas as pd

# Path to your specific database
db_path = r'C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\data\database.sqlite'


def get_table_schema(path, table_name):
    try:
        conn = sqlite3.connect(path)
        # Using PRAGMA to get metadata
        query = f"PRAGMA table_info({table_name});"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return f"No table found named '{table_name}'"

        # Rename columns for clarity
        df = df[['cid', 'name', 'type', 'notnull', 'pk']]
        return df
    except Exception as e:
        return f"Error: {e}"


# Execute and display
schema_df = get_table_schema(db_path, 'cleaned_scaled_data')
print(schema_df.to_string(index=False))