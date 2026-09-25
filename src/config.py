import os

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

BUCKET = os.getenv(
    "BUCKET",
    "kartheek-business-er-2026"
)

S3_ROOT = f"s3://{BUCKET}"

RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"
BLOCKING_PREFIX = "blocking"
FEATURE_PREFIX = "features"
MODEL_PREFIX = "models"
PREDICTION_PREFIX = "predictions"
OUTPUT_PREFIX = "output"


TRAIN_S1 = f"{S3_ROOT}/{RAW_PREFIX}/train/train_source1.tsv"
TRAIN_S2 = f"{S3_ROOT}/{RAW_PREFIX}/train/train_source2.tsv"
TRAIN_S3 = f"{S3_ROOT}/{RAW_PREFIX}/train/train_source3.tsv"

GROUND_TRUTH = (
    f"{S3_ROOT}/{RAW_PREFIX}/train/train_ground_truth.tsv"
)

TEST_S1 = f"{S3_ROOT}/{RAW_PREFIX}/test/test_source1.tsv"
TEST_S2 = f"{S3_ROOT}/{RAW_PREFIX}/test/test_source2.tsv"
TEST_S3 = f"{S3_ROOT}/{RAW_PREFIX}/test/test_source3.tsv"


FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "address_ratio",
    "address_partial_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "numeric_overlap",
    "country_match",
    "name_length_diff",
    "address_length_diff",
]
