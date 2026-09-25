import pandas as pd
import xgboost as xgb

from src.config import FEATURE_COLUMNS


def load_model(
    model_path,
):

    model = xgb.XGBClassifier()

    model.load_model(
        model_path
    )

    return model


def predict_features(
    model,
    features,
):

    probabilities = model.predict_proba(
        features[
            FEATURE_COLUMNS
        ]
    )[:, 1]

    result = features.copy()

    result[
        "match_probability"
    ] = probabilities

    return result
