import pandas as pd

from src.preprocessing import (
    normalize_business_name,
    normalize_address,
)


def test_business_name():

    result = normalize_business_name(
        "ABC Technologies Pvt. Ltd."
    )

    assert result == "abc technologies"


def test_address():

    result = normalize_address(
        "123 Main Street, Suite 5"
    )

    assert result == "123 main st ste 5"


if __name__ == "__main__":
    test_business_name()
    test_address()

    print(
        "Preprocessing tests passed."
    )
