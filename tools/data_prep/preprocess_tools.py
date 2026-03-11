import os

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine

from imblearn.over_sampling import SMOTE

from utils.visualizer import VisualizerFacade

import logging
from pathlib import Path

import pandas as pd
import joblib
import os
import numpy as np
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config.settings import settings

# Configure logger for terminal visibility
logger = logging.getLogger("Phase1_Runner")


def apply_scaling_and_cleaning_tool(db_path: str, target_col: str = 'is_fraud') -> str:
    try:
        # --- 1. SETUP ---
        current_file = Path(__file__).resolve()
        project_root = next(p for p in current_file.parents if p.name == 'mas_fraud_detector')
        plot_dir = os.path.join(project_root, "reports", "plots")
        visualizer = VisualizerFacade(output_dir=plot_dir)

        # --- 2. LOADING & RAW STATE LOG ---
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM train_transactions", conn)

        if df.empty: return "PREPROCESS ERROR: Table empty."

        logger.info(f"\n[RAW DATA PREVIEW]\n{df.head().to_string()}\n")
        visualizer.dist_plotter.plot_class_balance(df, plot_dir, "Pre-SMOTE")

        # --- 3. SCALING STATE LOG ---
        y = df[target_col]
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        X = X.drop(columns=[c for c in ['txn_id', 'Unnamed: 0', 'index'] if c in X.columns])

        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X)

        # Log the scaled state to verify normalization (Mean ~0, Std ~1)
        scaled_preview = pd.DataFrame(X_scaled_array[:5], columns=X.columns)
        logger.info(f"\n[SCALED FEATURE PREVIEW (Z-Score)]\n{scaled_preview.to_string()}\n")

        # Save Scaler
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

        # --- 4. SMOTE & FINAL STATE LOG ---
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled_array, y)

        cleaned_df = pd.DataFrame(X_resampled, columns=X.columns)
        cleaned_df[target_col] = y_resampled

        logger.info(f"\n[POST-SMOTE BALANCED PREVIEW]\n{cleaned_df.head().to_string()}\n")
        logger.info(f"New Class Distribution: {y_resampled.value_counts().to_dict()}")

        # Snapshot 2: Verify balance
        visualizer.verify_resampling(cleaned_df)

        # Save to SQL
        cleaned_df.to_sql("cleaned_scaled_data", engine, if_exists='replace', index=False)

        return (f"PREPROCESS SUCCESS: Data scaled, SMOTE applied. "
                f"Final Distribution: {len(cleaned_df)} rows.")

    except Exception as e:
        logger.error(f"PREPROCESS FAILURE: {str(e)}")
        return f"PREPROCESS ERROR: {str(e)}"


def drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs dimensionality reduction by removing non-predictive and
    high-cardinality features.

    Rationale for Drops:
    - Unique IDs: 'txn_id', 'cc_num', 'trans_num', 'Unnamed: 0', 'index'
      are removed to prevent the model from 'memorizing' specific
      transactions (overfitting).
    - Personal Identifiers: 'first', 'last', 'street' are dropped for
      privacy (PII) and because they have no statistical correlation
      with fraud patterns.
    - High-Cardinality Strings: 'merchant' and 'job' often contain
      hundreds of unique labels. Encoding these would result in
      extremely sparse matrices, degrading model convergence.
    - Temporal/Spatial Redundancy: 'trans_date_trans_time' and 'dob'
      are complex strings; their value is captured more efficiently
      by 'unix_time' (if kept) or coordinate data.
    - Geographic Labels: 'city' and 'state' are dropped because their
      predictive value is fully represented by numerical 'lat' and
      'long' features, which allow the model to learn spatial clusters.
    """
    drop_cols = [
        'txn_id', 'Unnamed: 0', 'index', 'cc_num', 'first', 'last',
        'street', 'trans_num', 'job', 'merchant', 'dob',
        'trans_date_trans_time', 'city', 'state'
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def handle_categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms categorical variables into a numerical format
    suitable for gradient-based and tree-based algorithms.

    Rationale for Techniques:
    - One-Hot Encoding (OHE): Applied to 'category' and 'gender'
      because they are nominal (no inherent order). OHE prevents
      the model from assuming 'Grocery' (e.g., 2) is 'greater than'
      'Entertainment' (e.g., 1).
    - Boolean-to-Integer Casting: pd.get_dummies often outputs
      bool types (True/False). Many ML frameworks (like XGBoost)
      require int/float (1/0) for tensor operations.
    - Label Encoding (Avoided): We avoided Label Encoding for
      unordered categories to prevent the model from learning
      false ordinal relationships.
    """
    # 1. Generate Dummy variables
    df = pd.get_dummies(df, columns=['category', 'gender'], drop_first=True)

    # 2. Cast Booleans to Integers (1/0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def scale_numerical_features(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    Standardizes the ENTIRE feature set (21 columns) so the scaler
    matches the model input dimensions exactly.
    """
    # 1. Identify all features except the target
    features_to_scale = [c for c in df.columns if c != 'is_fraud']

    # 2. Fit/Transform the entire 21-column block
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 3. Export the "21-column Gatekeeper"
    import joblib
    joblib.dump(scaler, scaler_path)

    # Also save the feature names so we never forget the order
    joblib.dump(features_to_scale, "models/feature_names.joblib")

    return df



def get_post_process_schema(df: pd.DataFrame) -> str:
    """Generates a clean Markdown table of the processed numeric data."""
    # Create a summary DataFrame
    summary = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": [len(df) - df[col].isnull().sum() for col in df.columns],
        "Dtype": [str(t) for t in df.dtypes]
    })

    return "### 📊 Post-Preprocessing Feature Schema\n" + summary.to_markdown(index=False)


def preprocessing_tool(db_path: str, target_col: str = 'is_fraud') -> str:
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM train_transactions", engine)

        # 1. CLEANING (Pandas) - Keep this for dropping PII/IDs
        df = drop_irrelevant_features(df)

        # 2. DEFINE GROUPS
        num_cols = ["amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
        cat_cols = ["category", "gender"]

        # 3. THE UNIFIED PREPROCESSOR (The Real OHE happens here)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ],
            remainder='drop'  # Ensures we only keep what we defined
        )

        # 4. FIT & TRANSFORM
        X_raw = df[num_cols + cat_cols]
        y = df[target_col].values

        X_out = preprocessor.fit_transform(X_raw)

        # 5. RECONSTRUCT WITH NAMES
        # This keeps your 22-24 features properly labeled in SQL
        feature_names = preprocessor.get_feature_names_out()
        final_df = pd.DataFrame(X_out, columns=feature_names)
        final_df[target_col] = y

        # 6. PERSISTENCE
        final_df.to_sql('cleaned_scaled_data', engine, if_exists='replace', index=False)

        # SAVE THE ENTIRE BRAIN (OHE + Scaler)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor_base.joblib"))

        return f"PREPROCESS_SUCCESS: Unified Sklearn Preprocessor Created. Total Features: {len(feature_names)}."

    except Exception as e:
        return f"PREPROCESS ERROR: {str(e)}"
#
def scrub_junk_columns(db_path: str):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db_path)
    # 1. Identify the 27 Gold Features + the Target
    gold_features = [
        'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'amt_to_cat_avg', 'high_risk_time', 'txn_velocity',
        'category_entertainment', 'category_food_dining', 'category_gas_transport',
        'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
        'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos',
        'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
        'category_travel', 'gender_F', 'gender_M', 'is_fraud'
    ]

    try:
        # 2. Get the current schema
        df_sample = pd.read_sql("SELECT * FROM cleaned_scaled_data LIMIT 1", conn)
        current_cols = df_sample.columns.tolist()

        # 3. Identify Junk (The 'Remainders')
        junk_cols = [c for c in current_cols if c not in gold_features]

        if junk_cols:
            print(f"🧹 Scrubbing {len(junk_cols)} junk columns: {junk_cols}")

            # SQLite doesn't support "DROP COLUMN" in older versions easily,
            # so the safest 'Universal' way is to recreate the table:
            query = f"CREATE TABLE cleaned_scaled_data_new AS SELECT {', '.join(gold_features)} FROM cleaned_scaled_data"
            conn.execute(query)
            conn.execute("DROP TABLE cleaned_scaled_data")
            conn.execute("ALTER TABLE cleaned_scaled_data_new RENAME TO cleaned_scaled_data")
            conn.execute("VACUUM")  # Reclaim space

            print("✅ Database scrubbed. 'cleaned_scaled_data' is now exactly 27 features + 1 target.")
        else:
            print("✨ No junk columns found. Schema is already clean.")

    except Exception as e:
        print(f"❌ Scrub failed: {e}")
    finally:
        conn.close()

