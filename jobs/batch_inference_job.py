import argparse
import os

import pandas as pd
import xgboost as xgb

from src.config import (
    FEATURE_COLUMNS,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_parquet(
        args.input
    )

    model = xgb.XGBClassifier()

    model.load_model(
        args.model
    )

    probabilities = model.predict_proba(
        df[
            FEATURE_COLUMNS
        ]
    )[:, 1]

    df[
        "match_probability"
    ] = probabilities

    os.makedirs(
        os.path.dirname(
            args.output
        ),
        exist_ok=True,
    )

    df.to_parquet(
        args.output,
        index=False,
    )

    print(
        f"Predictions: {len(df):,}"
    )


if __name__ == "__main__":
    main()
