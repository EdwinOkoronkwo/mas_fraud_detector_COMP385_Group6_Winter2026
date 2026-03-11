import os
import sqlite3

import joblib
import pandas as pd
import logging


class DataAuditor:
    def __init__(self, settings, logger=None):
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    def verify_physical_artifacts(self):
        """Checks for Phase 1 output artifacts only."""
        scaler_path = os.path.join(self.settings.MODELS_DIR, "scaler.joblib")
        # We check for the DB itself, not a balanced CSV
        checks = {
            "Database": self.settings.DB_PATH,
            "Scaler Object": scaler_path
        }

        self.logger.info("--- 🔎 PHASE 1 ARTIFACT AUDIT ---")
        results = {}
        for name, path in checks.items():
            exists = os.path.exists(path)
            results[name] = exists
            self.logger.info(f"{'✔' if exists else '✘'} {name}: {'FOUND' if exists else 'MISSING'}")
        return results

    def run_database_inspection(self):
        """Deep dive into SQLite tables and class distributions."""
        self.logger.info("📊 STARTING DATABASE INSPECTION...")
        try:
            with sqlite3.connect(self.settings.DB_PATH) as conn:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

                for table in tables['name']:
                    self.logger.info(f"--- Table: {table} ---")
                    count = pd.read_sql_query(f"SELECT COUNT(*) as total FROM {table}", conn)
                    self.logger.info(f"   Count: {count['total'][0]:,}")

                    try:
                        balance = pd.read_sql_query(
                            f"SELECT is_fraud, COUNT(*) as count FROM {table} GROUP BY is_fraud", conn)
                        self.logger.info(f"   Distribution:\n{balance.to_string(index=False)}")
                    except:
                        self.logger.info("   (is_fraud column not in this table)")
        except Exception as e:
            self.logger.error(f"💥 Inspection Failed: {e}")

    def verify_sql_handoff(self):
        """
        FINAL QUALITY GATE:
        Verifies the SQL table 'cleaned_scaled_data' exists and is model-ready.
        This ensures Phase 2 receives exactly what Phase 1 produced.
        """
        self.logger.info("🔍 PHASE 2 INPUT VERIFICATION (SQL DIRECT)")

        # 1. Replicate the Phase 2 connection logic exactly
        db_path = os.path.abspath(self.settings.DB_PATH)
        try:
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{db_path}")

            # 2. Test the query Phase 2 will use
            df = pd.read_sql("SELECT * FROM cleaned_scaled_data LIMIT 5", engine)

            # 3. Log the Path and the Data Head
            self.logger.info(f"📍 Database Input: {db_path}")
            self.logger.info("📋 TABLE PREVIEW ('cleaned_scaled_data'):")
            self.logger.info(f"\n{df.to_string(index=False)}")

            # 4. Integrity Checks
            row_count = pd.read_sql("SELECT COUNT(*) as count FROM cleaned_scaled_data", engine).iloc[0]['count']
            self.logger.info(f"📊 Total Rows Available: {row_count}")

            # Ensure 'is_fraud' exists (Phase 2's target)
            if 'is_fraud' in df.columns:
                self.logger.info("✅ Target column 'is_fraud' found.")
            else:
                self.logger.error("❌ FAIL: Target column 'is_fraud' is missing!")
                return False

            return True

        except Exception as e:
            self.logger.error(f"💥 SQL HANDOFF ERROR: {e}")
            return False