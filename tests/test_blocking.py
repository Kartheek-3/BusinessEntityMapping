import pandas as pd

from src.preprocessing import (
    normalize_dataframe,
)

from src.blocking import (
    build_all_indexes,
    generate_candidates,
)


def test_blocking():

    s1 = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name":
                "ABC Technologies Pvt Ltd",
            "business_address":
                "123 Main Street",
            "country": "US",
        }
    ])

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-1",
            "business_name":
                "ABC Technologies",
            "business_address":
                "123 Main St",
            "country": "US",
        }
    ])

    s3 = pd.DataFrame([
        {
            "entity_id": "S3-1",
            "business_name":
                "XYZ Company",
            "business_address":
                "500 Oak Rd",
            "country": "US",
        }
    ])

    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)

    indexes_s2 = build_all_indexes(s2)
    indexes_s3 = build_all_indexes(s3)

    candidates = generate_candidates(
        s1.iloc[0],
        s2,
        s3,
        indexes_s2,
        indexes_s3,
    )

    assert "S2-1" in candidates


if __name__ == "__main__":
    test_blocking()

    print(
        "Blocking tests passed."
    )
