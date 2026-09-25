import argparse
import os

import pandas as pd

from src.features import (
    create_features,
)


def load_ground_truth(
    path,
):

    if not path:
        return {}

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    truth = {}

    for _, row in gt.iterrows():

        value = row.get(
            "matched_entity_ids",
            "",
        )

        if pd.isna(value):
            value = ""

        truth[
            row["source1_entity_id"]
        ] = set(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )

    return truth


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
        "--candidates",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
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

    candidates = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
    )

    truth = load_ground_truth(
        args.ground_truth
    )

    s1_lookup = s1.set_index(
        "entity_id"
    ).to_dict("index")

    s2_lookup = s2.set_index(
        "entity_id"
    ).to_dict("index")

    s3_lookup = s3.set_index(
        "entity_id"
    ).to_dict("index")

    rows = []

    for _, candidate_row in candidates.iterrows():

        s1_id = candidate_row[
            "source1_entity_id"
        ]

        source1 = s1_lookup.get(
            s1_id
        )

        if source1 is None:
            continue

        value = candidate_row.get(
            "candidate_entity_ids",
            "",
        )

        if pd.isna(value):
            continue

        for candidate_id in value.split(","):

            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            if candidate_id.startswith("S2-"):
                candidate = s2_lookup.get(
                    candidate_id
                )
            elif candidate_id.startswith("S3-"):
                candidate = s3_lookup.get(
                    candidate_id
                )
            else:
                continue

            if candidate is None:
                continue

            features = create_features(
                source1,
                candidate,
            )

            features[
                "source1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = candidate_id

            if truth:

                features["label"] = int(
                    candidate_id
                    in truth.get(
                        s1_id,
                        set(),
                    )
                )

            rows.append(
                features
            )

    output = pd.DataFrame(
        rows
    )

    os.makedirs(
        os.path.dirname(
            args.output
        ),
        exist_ok=True,
    )

    output.to_parquet(
        args.output,
        index=False,
    )

    print(
        f"Feature rows: {len(output):,}"
    )


if __name__ == "__main__":
    main()
