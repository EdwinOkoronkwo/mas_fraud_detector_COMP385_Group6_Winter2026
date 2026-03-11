import sqlite3
import pandas as pd


class DataHandler:
    """Handles all database interactions for the inference pipeline.

    This class is responsible for connecting to the transaction database and
    retrieving balanced datasets to ensure models are tested against both
    fraudulent and legitimate examples.
    """

    def __init__(self, db_path: str):
        """Initializes the DataHandler with the path to the SQLite database.

        Args:
            db_path: The absolute or relative path to the .db file.
        """
        self.db_path = db_path

    def fetch_balanced_samples(self, n_samples: int) -> pd.DataFrame:
        limit_per_class = n_samples // 2  # Ensures 5 fraud and 5 normal if n=10
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT * \
                FROM (SELECT CAST(cc_num AS TEXT) as cc_num, \
                             amt, \
                             zip, \
                             lat, \
                             long, \
                             city_pop, \
                             unix_time, \
                             merch_lat, \
                             merch_long, \
                             category, \
                             gender, \
                             is_fraud             as actual_label \
                      FROM test_transactions \
                      WHERE is_fraud = 1 LIMIT ?)
                UNION ALL
                SELECT * \
                FROM (SELECT CAST(cc_num AS TEXT) as cc_num, \
                             amt, \
                             zip, \
                             lat, \
                             long, \
                             city_pop, \
                             unix_time, \
                             merch_lat, \
                             merch_long, \
                             category, \
                             gender, \
                             is_fraud             as actual_label \
                      FROM test_transactions \
                      WHERE is_fraud = 0 LIMIT ?)
                """
        df = pd.read_sql_query(query, conn, params=[limit_per_class, limit_per_class])
        conn.close()

        # Shuffle and reset index so the loop runs 0 to 9 cleanly
        return df.sample(frac=1).reset_index(drop=True)

    # def fetch_balanced_samples(self, n_samples: int) -> pd.DataFrame:
    #     """Retrieves an equal number of fraud and non-fraud transactions.
    #
    #     Args:
    #         n_samples: The number of transactions to pull for each category
    #                    (e.g., if 10, returns 20 total rows).
    #
    #     Returns:
    #         A pandas DataFrame containing transaction features and actual labels.
    #     """
    #     conn = sqlite3.connect(self.db_path)
    #     query = """
    #         SELECT * FROM (
    #             SELECT CAST(cc_num AS TEXT) as cc_num, amt, zip, lat, long,
    #                    city_pop, unix_time, merch_lat, merch_long, is_fraud as actual_label
    #             FROM train_transactions WHERE is_fraud = 1 LIMIT ?
    #         )
    #         UNION ALL
    #         SELECT * FROM (
    #             SELECT CAST(cc_num AS TEXT) as cc_num, amt, zip, lat, long,
    #                    city_pop, unix_time, merch_lat, merch_long, is_fraud as actual_label
    #             FROM train_transactions WHERE is_fraud = 0 LIMIT ?
    #         )
    #     """
    #     df = pd.read_sql_query(query, conn, params=[n_samples, n_samples])
    #     conn.close()
    #     return df