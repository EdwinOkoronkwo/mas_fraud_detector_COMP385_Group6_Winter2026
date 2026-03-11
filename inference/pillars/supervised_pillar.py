import joblib
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger("Supervised_Pillar")

import joblib
import pandas as pd
import numpy as np

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np

import joblib
import pandas as pd
import xgboost as xgb


# inference/pillars/supervised_pillar.py

# inference/pillars/supervised_pillar.py

import joblib
import pandas as pd
import xgboost as xgb

import pandas as pd
import joblib
import xgboost as xgb


class SupervisedPillar:
    def __init__(self, model_path, feature_list):
        self.model_path = model_path
        self.feature_list = feature_list
        self.model = joblib.load(model_path)
        self.is_sklearn = hasattr(self.model, 'predict_proba')

        # Memory storage for performance tracking
        self.last_prediction = None
        self.last_probability = None

    def predict(self, input_dict: dict) -> float:
        df = pd.DataFrame([input_dict])
        final_input = df.reindex(columns=self.feature_list, fill_value=0.0)
        final_input = final_input.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        if self.is_sklearn:
            prob = float(self.model.predict_proba(final_input)[0][1])
        else:
            dmatrix = xgb.DMatrix(final_input, feature_names=self.feature_list)
            prob = float(self.model.predict(dmatrix)[0])

        # STORE THE RESULTS LOCALLY
        self.last_probability = prob
        self.last_prediction = 1 if prob > 0.5 else 0

        return prob