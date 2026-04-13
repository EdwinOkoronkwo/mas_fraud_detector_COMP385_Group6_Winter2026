# api/controllers/inference_controller.py
import asyncio

import pandas as pd

from data.database import SessionLocal
from data.models.customer import Customer
from data.models.inference_result import InferenceResult

import asyncio
from data.database import SessionLocal
from data.models.inference_result import InferenceResult
from deterministic_inference.utils.geo_math import calculate_dist


class InferenceController:
    def __init__(self, pipeline):
        """
        :param pipeline: The InferencePipeline instance
        """
        self.pipeline = pipeline

    async def run_ui_batch(self, n_samples: int, params: dict = None, callback=None):
        # 1. Execute the Batch
        results_df = await self.pipeline.run_batch(
            n_samples=n_samples,
            params=params,
            callback=callback,
        )

        # 2. Database Sync
        db = SessionLocal()
        try:
            if callback:
                callback("SYSTEM", f"Syncing {len(results_df)} records to Compliance Database...")

            # Inside InferenceController.run_ui_batch
            for _, row in results_df.iterrows():
                record = InferenceResult(
                    # Standardize: Processors often use 'CC', DB uses 'cc_num'
                    cc_num=str(row.get('CC', row.get('cc_num', ''))),
                    merchant=row.get('merchant', 'Unknown'),

                    # Identity Logic (Crucial for Nathan Massey)
                    # We store these if your DB model has first_name/last_name columns
                    # If not, we ensure they exist in results_df for the UI to see them
                    actual_label=int(row.get('ACT', 0)),

                    # Pillars
                    gold_score=float(row.get('GOLD', 0.0)),
                    neuro_raw=float(row.get('N_RAW', 0.0)),
                    neuro_norm=float(row.get('N_CAL', 0.0)),  # Match the key from package_row
                    cluster_raw=float(row.get('C_RAW', 0.0)),
                    cluster_norm=float(row.get('C_CAL', 0.0)),

                    # Results
                    final_mas_score=float(row.get('MATH', 0.0)),
                    prediction_mode=row.get('mode', 'N/A'),
                    explanation=row.get('explanation', ''),
                )
                db.add(record)
            db.commit()
            if callback:
                callback("SYSTEM", "✅ Database Sync Complete.")

        except Exception as e:
            print(f"❌ Sync Failure: {e}")
            db.rollback()
            raise e
        finally:
            db.close()

        return results_df

    # Inside InferenceController
    async def run_single_inference(self, tx_data: dict, params: dict, callback=None):
        """
        Diagnostic Version: Logs the exact shape of the data to the terminal.
        """
        df_single = pd.DataFrame([tx_data])
        if 'Unnamed: 0' not in df_single.columns:
            df_single['Unnamed: 0'] = 0

        try:
            result_output = await self.pipeline.run_batch(
                n_samples=1,
                manual_df=df_single,
                params=params,
                callback=callback
            )

            # 🔍 THE DIAGNOSTIC LOGS
            print("\n--- DEBUG: SINGLE AUDIT DATA TYPE ---")
            print(f"Object Type: {type(result_output)}")
            print(f"Content: {result_output}")
            print("-------------------------------------\n")

            # 🚀 SMART HANDLING BASED ON LOGS
            if isinstance(result_output, pd.DataFrame):
                return result_output.iloc[0].to_dict()

            elif isinstance(result_output, list):
                # If it's a list, the first item is usually our result dict
                return result_output[0] if len(result_output) > 0 else {}

            elif isinstance(result_output, dict):
                return result_output

            return result_output

        except Exception as e:
            print(f"❌ Diagnostic Error: {e}")
            raise e

    def get_transaction_by_cc(self, cc_num: str):
        """Bridge between the Single Audit View and the DataHandler."""
        # 🚀 FIX: Call the new method we just added to the handler
        df_match = self.pipeline.handler.get_transaction_by_cc(cc_num)

        if not df_match.empty:
            # Convert the single-row DataFrame to a dictionary for the RAG prompt
            return df_match.iloc[0].to_dict()

        return None