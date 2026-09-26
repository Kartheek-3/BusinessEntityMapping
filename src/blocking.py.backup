from collections import defaultdict


def get_name_tokens(name):
    if not name:
        return []

    return [
        token
        for token in name.split()
        if len(token) >= 3
    ]


def build_name_index(df):
    index = defaultdict(list)

    for idx, name in enumerate(df["name_norm"]):
        if name:
            index[name].append(idx)

    return index


def build_prefix_index(
    df,
    prefix_length=4,
):
    index = defaultdict(list)

    for idx, row in df.iterrows():

        name = row["name_norm"]
        country = row["country_norm"]

        if not name:
            continue

        key = (
            country,
            name[:prefix_length],
        )

        index[key].append(idx)

    return index


def build_token_index(df):
    index = defaultdict(list)

    for idx, name in enumerate(
        df["name_norm"]
    ):

        for token in get_name_tokens(name):
            index[token].append(idx)

    return index


def build_all_indexes(df):
    return {
        "name": build_name_index(df),
        "prefix": build_prefix_index(df),
        "token": build_token_index(df),
    }


def generate_candidates(
    s1_row,
    s2,
    s3,
    indexes_s2,
    indexes_s3,
    max_token_candidates=200,
):
    candidates = set()

    name = s1_row["name_norm"]
    country = s1_row["country_norm"]

    # Exact normalized name
    if name:

        for idx in indexes_s2["name"].get(
            name,
            [],
        ):
            candidates.add(
                s2.iloc[idx]["entity_id"]
            )

        for idx in indexes_s3["name"].get(
            name,
            [],
        ):
            candidates.add(
                s3.iloc[idx]["entity_id"]
            )

    # Country + prefix
    if name:

        key = (
            country,
            name[:4],
        )

        for idx in indexes_s2["prefix"].get(
            key,
            [],
        ):
            candidates.add(
                s2.iloc[idx]["entity_id"]
            )

        for idx in indexes_s3["prefix"].get(
            key,
            [],
        ):
            candidates.add(
                s3.iloc[idx]["entity_id"]
            )

    # Token blocking
    token_candidates = set()

    for token in get_name_tokens(name):

        token_candidates.update(
            indexes_s2["token"].get(
                token,
                [],
            )
        )

        if len(token_candidates) >= max_token_candidates:
            break

    for token in get_name_tokens(name):

        if len(token_candidates) >= max_token_candidates:
            break

        token_candidates.update(
            indexes_s3["token"].get(
                token,
                [],
            )
        )

    for idx in list(token_candidates)[
        :max_token_candidates
    ]:

        if idx < len(s2):

            candidates.add(
                s2.iloc[idx]["entity_id"]
            )

        else:

            s3_idx = idx - len(s2)

            if s3_idx < len(s3):

                candidates.add(
                    s3.iloc[s3_idx]["entity_id"]
                )

    return candidates
