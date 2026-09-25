from src.features import (
    create_features,
)


def test_features():

    source1 = {
        "name_norm": "abc technologies",
        "address_norm":
            "123 main st ste 5",
        "country_norm": "us",
    }

    candidate = {
        "name_norm": "abc technologies",
        "address_norm":
            "123 main st suite 5",
        "country_norm": "us",
    }

    features = create_features(
        source1,
        candidate,
    )

    assert (
        features["name_ratio"] == 1.0
    )

    assert (
        features["country_match"] == 1
    )


if __name__ == "__main__":
    test_features()

    print(
        "Feature tests passed."
    )
