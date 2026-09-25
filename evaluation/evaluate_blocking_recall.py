import argparse

import pandas as pd


def parse_truth(value):

    if pd.isna(value):
        return set()

    return set(
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--candidates",
        required=True,
    )

    args = parser.parse_args()

    gt = pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype=str,
    )

    candidates = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
    )

    candidate_map = {}

    for _, row in candidates.iterrows():

        value = row.get(
            "candidate_entity_ids",
            "",
        )

        candidate_map[
            row["source1_entity_id"]
        ] = set(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )

    total_true = 0
    found_true = 0

    entities_with_truth = 0
    entities_all_found = 0

    for _, row in gt.iterrows():

        truth = parse_truth(
            row["matched_entity_ids"]
        )

        if not truth:
            continue

        entities_with_truth += 1

        candidate_set = candidate_map.get(
            row["source1_entity_id"],
            set(),
        )

        intersection = (
            truth & candidate_set
        )

        total_true += len(truth)

        found_true += len(
            intersection
        )

        if truth.issubset(
            candidate_set
        ):
            entities_all_found += 1

    pair_recall = (
        found_true / total_true
        if total_true
        else 0.0
    )

    entity_recall = (
        entities_all_found
        / entities_with_truth
        if entities_with_truth
        else 0.0
    )

    print(
        f"Pair candidate recall: "
        f"{pair_recall:.4%}"
    )

    print(
        f"All-match entity recall: "
        f"{entity_recall:.4%}"
    )


if __name__ == "__main__":
    main()
