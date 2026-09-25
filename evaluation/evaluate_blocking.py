import argparse

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
    )

    counts = (
        df[
            "candidate_entity_ids"
        ]
        .fillna("")
        .apply(
            lambda x:
                0
                if not x
                else len(
                    set(
                        item.strip()
                        for item in x.split(",")
                        if item.strip()
                    )
                )
        )
    )

    print(
        f"Entities: {len(df):,}"
    )

    print(
        f"Average candidates: {counts.mean():.2f}"
    )

    print(
        f"Maximum candidates: {counts.max():,}"
    )

    print(
        f"Zero candidates: {(counts == 0).sum():,}"
    )


if __name__ == "__main__":
    main()
