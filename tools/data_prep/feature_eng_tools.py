import pandas as pd
import numpy as np


class FraudToolbox:
    """Atomic Python functions for fraud feature induction."""

    @staticmethod
    def compute_amt_to_cat_ratio(df, category_means):
        """Calculates how much a transaction deviates from the category norm."""
        # Fallback to global mean if category is missing
        global_avg = sum(category_means.values()) / len(category_means)
        df['amt_to_cat_avg'] = df.apply(
            lambda x: x['amt'] / category_means.get(x['category'], global_avg), axis=1
        )
        return df

    @staticmethod
    def extract_temporal_risk(df):
        """Flags transactions in high-risk 'dead of night' windows."""
        # Extracts hour from unix_time
        hours = pd.to_datetime(df['unix_time'], unit='s').dt.hour
        df['high_risk_time'] = hours.apply(lambda x: 1 if x <= 4 or x >= 23 else 0)
        return df

    @staticmethod
    def calculate_velocity(df, window='24h'):
        """Computes transaction frequency per cardholder."""
        # Simple count per cc_num for the current batch
        df['txn_velocity'] = df.groupby('cc_num')['amt'].transform('count')
        return df