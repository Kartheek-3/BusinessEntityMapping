import argparse
import os

import pandas as pd
import xgboost as xgb

from src.config import FEATURE_COLUMNS


def train_model(
    train_path,
    model_dir,
):

    df = pd.read_parquet(
        train_path
    )

    if "label" not in df.columns:
        raise ValueError(
            "Training dataset must contain 'label'"
        )

    X = df[
        FEATURE_COLUMNS
    ]

    y = df["label"].astype(int)

    positive = int(
        y.sum()
    )

    negative = int(
        len(y) - positive
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    model.save_model(
        os.path.join(
            model_dir,
            "model.json",
        )
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        default="models",
    )

    args = parser.parse_args()

    train_model(
        args.train,
        args.model_dir,
    )


if __name__ == "__main__":
    main()
