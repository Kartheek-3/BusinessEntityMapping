import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


# ============================================================
# ARGUMENTS
# ============================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "INPUT_PATH",
        "OUTPUT_PATH",
    ],
)

INPUT_PATH = args["INPUT_PATH"]
OUTPUT_PATH = args["OUTPUT_PATH"]


# ============================================================
# SPARK / GLUE INITIALIZATION
# ============================================================

spark = (
    SparkSession.builder
    .appName("BusinessEntityResolution-Preprocessing")
    .getOrCreate()
)

glue_context = GlueContext(spark.sparkContext)
job = Job(glue_context)
job.init(args["JOB_NAME"], args)


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(column):
    """
    General text normalization:
    - null -> empty string
    - lowercase
    - replace punctuation with spaces
    - collapse multiple spaces
    - trim
    """

    result = F.coalesce(column, F.lit(""))

    result = F.lower(result)

    # Replace punctuation / special characters with spaces.
    result = F.regexp_replace(
        result,
        r"[^a-z0-9\s]",
        " "
    )

    # Collapse multiple spaces.
    result = F.regexp_replace(
        result,
        r"\s+",
        " "
    )

    return F.trim(result)


def normalize_business_name(column):
    """
    Business-name normalization.

    Examples:
        ABC Pvt Ltd       -> abc
        ABC PRIVATE LTD   -> abc
        ABC, Inc.         -> abc
        ABC Technologies  -> abc technologies

    Legal suffixes are removed only when they occur
    at the end of the normalized business name.
    """

    result = normalize_text(column)

    # Remove common legal suffixes from the END.
    legal_suffix_pattern = (
        r"\s+"
        r"(private\s+limited|"
        r"pvt\s+limited|"
        r"pvt\s+ltd|"
        r"private\s+ltd|"
        r"limited|"
        r"ltd|"
        r"incorporated|"
        r"inc|"
        r"corporation|"
        r"corp|"
        r"llc|"
        r"llp|"
        r"plc|"
        r"co|"
        r"company)"
        r"$"
    )

    result = F.regexp_replace(
        result,
        legal_suffix_pattern,
        ""
    )

    result = F.regexp_replace(
        result,
        r"\s+",
        " "
    )

    return F.trim(result)


def normalize_address(column):
    """
    Address normalization.

    Handles common address abbreviations such as:

        street -> st
        road -> rd
        avenue -> ave
        boulevard -> blvd
        apartment -> apt
        suite -> ste
        highway -> hwy
    """

    result = normalize_text(column)

    replacements = [
        (r"\bstreet\b", "st"),
        (r"\broad\b", "rd"),
        (r"\bavenue\b", "ave"),
        (r"\bav\b", "ave"),
        (r"\bboulevard\b", "blvd"),
        (r"\bdrive\b", "dr"),
        (r"\blane\b", "ln"),
        (r"\bhighway\b", "hwy"),
        (r"\bparkway\b", "pkwy"),
        (r"\bcircle\b", "cir"),
        (r"\bcourt\b", "ct"),
        (r"\bplace\b", "pl"),
        (r"\bterrace\b", "ter"),
        (r"\bapartment\b", "apt"),
        (r"\bsuite\b", "ste"),
        (r"\bbuilding\b", "bldg"),
        (r"\bfloor\b", "fl"),
        (r"\broad\b", "rd"),
    ]

    for pattern, replacement in replacements:
        result = F.regexp_replace(
            result,
            pattern,
            replacement
        )

    result = F.regexp_replace(
        result,
        r"\s+",
        " "
    )

    return F.trim(result)


def normalize_country(column):
    """
    Country normalization.

    Does not hardcode the countries used by train/test.
    This is important because France appears in test data.
    """

    result = F.coalesce(column, F.lit(""))

    result = F.lower(result)

    result = F.regexp_replace(
        result,
        r"[^a-z0-9\s]",
        " "
    )

    result = F.regexp_replace(
        result,
        r"\s+",
        " "
    )

    return F.trim(result)


# ============================================================
# READ INPUT
# ============================================================

print("=" * 70)
print("BUSINESS ENTITY RESOLUTION - PREPROCESSING")
print("=" * 70)

print(f"Input : {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

print("\nReading TSV data...")

df = (
    spark.read
    .option("header", True)
    .option("sep", "\t")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", False)
    .option("inferSchema", False)
    .csv(INPUT_PATH)
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\nInput columns:")
print(df.columns)

print("\nInput schema:")
df.printSchema()


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

df = df.select(
    F.col("entity_id").cast("string"),
    F.col("business_name").cast("string"),
    F.col("business_address").cast("string"),
    F.col("country").cast("string"),
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nCreating normalized columns...")

df = (
    df
    .withColumn(
        "name_norm",
        normalize_business_name(
            F.col("business_name")
        )
    )
    .withColumn(
        "address_norm",
        normalize_address(
            F.col("business_address")
        )
    )
    .withColumn(
        "country_norm",
        normalize_country(
            F.col("country")
        )
    )
)


# ============================================================
# ADD BLOCKING HELPER COLUMNS
# ============================================================

print("\nCreating blocking helper columns...")

df = (
    df
    .withColumn(
        "name_prefix4",
        F.substring(
            F.col("name_norm"),
            1,
            4
        )
    )
    .withColumn(
        "name_prefix6",
        F.substring(
            F.col("name_norm"),
            1,
            6
        )
    )
)


# ============================================================
# DATA QUALITY STATISTICS
# ============================================================

print("\nCalculating statistics...")

row_count = df.count()

print(f"\nTotal rows: {row_count:,}")

empty_name_count = (
    df
    .filter(
        (F.col("name_norm") == "") |
        F.col("name_norm").isNull()
    )
    .count()
)

empty_address_count = (
    df
    .filter(
        (F.col("address_norm") == "") |
        F.col("address_norm").isNull()
    )
    .count()
)

empty_country_count = (
    df
    .filter(
        (F.col("country_norm") == "") |
        F.col("country_norm").isNull()
    )
    .count()
)

print(f"Empty normalized names    : {empty_name_count:,}")
print(f"Empty normalized addresses: {empty_address_count:,}")
print(f"Empty normalized countries: {empty_country_count:,}")


# ============================================================
# SAMPLE
# ============================================================

print("\nSample normalized records:")

(
    df
    .select(
        "entity_id",
        "business_name",
        "name_norm",
        "business_address",
        "address_norm",
        "country",
        "country_norm",
    )
    .show(10, truncate=False)
)


# ============================================================
# WRITE PARQUET
# ============================================================

print("\nWriting Parquet output...")

(
    df
    .repartition(8)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUTPUT_PATH)
)

print("\nOutput written successfully.")
print(f"Output location: {OUTPUT_PATH}")


# ============================================================
# FINISH
# ============================================================

job.commit()

print("=" * 70)
print("PREPROCESSING JOB COMPLETED")
print("=" * 70)