import argparse
import os

import pandas as pd

from src.blocking import (
    build_all_indexes,
    generate_candidates,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        required=True,
    )

    parser.add_argument(
        "--s2",
        required=True,
    )

    parser.add_argument(
        "--s3",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    s1 = pd.read_parquet(
        args.s1
    )

    s2 = pd.read_parquet(
        args.s2
    )

    s3 = pd.read_parquet(
        args.s3
    )

    print(
        "Building S2 indexes..."
    )

    indexes_s2 = build_all_indexes(
        s2
    )

    print(
        "Building S3 indexes..."
    )

    indexes_s3 = build_all_indexes(
        s3
    )

    rows = []

    for counter, (_, row) in enumerate(
        s1.iterrows(),
        start=1,
    ):

        candidates = generate_candidates(
            row,
            s2,
            s3,
            indexes_s2,
            indexes_s3,
        )

        rows.append({
            "source1_entity_id":
                row["entity_id"],

            "candidate_entity_ids":
                ",".join(
                    sorted(candidates)
                ),
        })

        if counter % 10000 == 0:
            print(
                f"Processed {counter:,} S1 entities"
            )

    result = pd.DataFrame(
        rows
    )

    os.makedirs(
        os.path.dirname(
            args.output
        ),
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    print(
        f"Generated {len(result):,} candidate rows"
    )


if __name__ == "__main__":
    main()
