import argparse
import os

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--candidates",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()

    predictions = pd.read_parquet(
        args.predictions
    )

    candidates = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
    )

    matches = predictions[
        predictions[
            "match_probability"
        ] >= args.threshold
    ]

    matched = (
        matches
        .groupby(
            "source1_entity_id"
        )["candidate_entity_id"]
        .apply(
            lambda values:
                ",".join(
                    sorted(
                        set(values)
                    )
                )
        )
        .to_dict()
    )

    results = pd.DataFrame({
        "source1_entity_id":
            candidates[
                "source1_entity_id"
            ].drop_duplicates()
    })

    results[
        "matched_entity_ids"
    ] = (
        results[
            "source1_entity_id"
        ]
        .map(matched)
        .fillna("")
    )

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    results.to_csv(
        os.path.join(
            args.output,
            "matching_results.tsv",
        ),
        sep="\t",
        index=False,
    )

    candidates.to_csv(
        os.path.join(
            args.output,
            "candidate_pairs.tsv",
        ),
        sep="\t",
        index=False,
    )

    print(
        "Final outputs generated."
    )


if __name__ == "__main__":
    main()
