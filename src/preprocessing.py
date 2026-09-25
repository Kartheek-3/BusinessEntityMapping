import re
import pandas as pd


LEGAL_SUFFIXES = [
    "private limited",
    "private ltd",
    "pvt limited",
    "pvt ltd",
    "pvt",
    "limited",
    "ltd",
    "llc",
    "incorporated",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
]


ADDRESS_REPLACEMENTS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "highway": "hwy",
    "hwy.": "hwy",
    "lane": "ln",
    "ln.": "ln",
    "drive": "dr",
    "dr.": "dr",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", " and ")

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_business_name(value):
    text = normalize_text(value)

    for suffix in LEGAL_SUFFIXES:
        text = re.sub(
            rf"\b{re.escape(suffix)}\b",
            " ",
            text,
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_address(value):
    text = normalize_text(value)

    for old, new in ADDRESS_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text,
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_country(value):
    return normalize_text(value)


def normalize_dataframe(df):
    df = df.copy()

    required = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
    )

    df["name_norm"] = (
        df["business_name"]
        .apply(normalize_business_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["country_norm"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df
