import argparse
import os

import pandas as pd

from src.preprocessing import (
    normalize_dataframe,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print(
        f"Reading {args.input}"
    )

    df = pd.read_csv(
        args.input,
        sep="\t",
        dtype=str,
    )

    print(
        f"Rows: {len(df):,}"
    )

    df = normalize_dataframe(
        df
    )

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
        f"Written {args.output}"
    )


if __name__ == "__main__":
    main()
