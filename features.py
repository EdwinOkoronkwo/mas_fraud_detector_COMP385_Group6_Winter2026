import joblib

# Load your specific preprocessor
preprocessor = joblib.load('models/preprocessor_base.joblib') # Adjust path if needed

if hasattr(preprocessor, 'feature_names_in_'):
    print("📋 EXACT COLUMNS REQUIRED:")
    print(preprocessor.feature_names_in_.tolist())
else:
    # If it's an older version or wrapped, we check the transformers
    print("📋 COLUMNS FROM TRANSFORMERS:")
    cols = []
    for name, transformer, columns in preprocessor.transformers_:
        if name != 'remainder':
            cols.extend(columns)
    print(cols)