import re

from rapidfuzz import fuzz


def safe_text(value):
    if value is None:
        return ""

    return str(value)


def jaccard_similarity(a, b):

    a_tokens = set(
        safe_text(a).split()
    )

    b_tokens = set(
        safe_text(b).split()
    )

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(
        a_tokens & b_tokens
    ) / len(
        a_tokens | b_tokens
    )


def extract_numbers(text):

    return set(
        re.findall(
            r"\d+",
            safe_text(text),
        )
    )


def numeric_overlap(a, b):

    a_numbers = extract_numbers(a)
    b_numbers = extract_numbers(b)

    if not a_numbers or not b_numbers:
        return 0.0

    return len(
        a_numbers & b_numbers
    ) / len(
        a_numbers | b_numbers
    )


def create_features(
    source1,
    candidate,
):

    name_a = safe_text(
        source1["name_norm"]
    )

    name_b = safe_text(
        candidate["name_norm"]
    )

    address_a = safe_text(
        source1["address_norm"]
    )

    address_b = safe_text(
        candidate["address_norm"]
    )

    return {
        "name_ratio": fuzz.ratio(
            name_a,
            name_b,
        ) / 100.0,

        "name_partial_ratio": fuzz.partial_ratio(
            name_a,
            name_b,
        ) / 100.0,

        "name_token_set_ratio": fuzz.token_set_ratio(
            name_a,
            name_b,
        ) / 100.0,

        "name_jaccard": jaccard_similarity(
            name_a,
            name_b,
        ),

        "address_ratio": fuzz.ratio(
            address_a,
            address_b,
        ) / 100.0,

        "address_partial_ratio": fuzz.partial_ratio(
            address_a,
            address_b,
        ) / 100.0,

        "address_token_set_ratio": fuzz.token_set_ratio(
            address_a,
            address_b,
        ) / 100.0,

        "address_jaccard": jaccard_similarity(
            address_a,
            address_b,
        ),

        "numeric_overlap": numeric_overlap(
            address_a,
            address_b,
        ),

        "country_match": int(
            source1["country_norm"]
            != ""
            and
            source1["country_norm"]
            ==
            candidate["country_norm"]
        ),

        "name_length_diff": abs(
            len(name_a)
            -
            len(name_b)
        ),

        "address_length_diff": abs(
            len(address_a)
            -
            len(address_b)
        ),
    }
