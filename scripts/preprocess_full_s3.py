import argparse
import os
import tempfile

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing import normalize_dataframe


# ============================================================
# CONFIG
# ============================================================

BUCKET = "kartheek-business-er-2026"
CHUNK_SIZE = 50_000

s3 = boto3.client("s3")


# ============================================================
# HELPERS
# ============================================================

def parse_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    value = uri[5:]
    bucket, key = value.split("/", 1)

    return bucket, key


def process_file(input_s3, output_s3):
    """
    Stream a large TSV from S3 in chunks, normalize each chunk,
    write incrementally to a local Parquet file, then upload it
    to S3.

    The complete TSV is never loaded into memory.
    """

    input_bucket, input_key = parse_s3_uri(input_s3)
    output_bucket, output_key = parse_s3_uri(output_s3)

    print("=" * 70)
    print("FULL DATA PREPROCESSING")
    print("=" * 70)
    print(f"Input : {input_s3}")
    print(f"Output: {output_s3}")
    print(f"Chunk : {CHUNK_SIZE:,} rows")
    print()

    print("Opening S3 object...")

    response = s3.get_object(
        Bucket=input_bucket,
        Key=input_key,
    )

    body = response["Body"]

    # pandas can stream from the S3 response body.
    reader = pd.read_csv(
        body,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        na_filter=False,
    )

    local_file = tempfile.NamedTemporaryFile(
        suffix=".parquet",
        delete=False,
    )

    local_path = local_file.name
    local_file.close()

    parquet_writer = None

    total_rows = 0
    chunk_number = 0

    try:

        for chunk in reader:

            chunk_number += 1

            print(
                f"Processing chunk {chunk_number:,} "
                f"({len(chunk):,} rows)..."
            )

            # ------------------------------------------------
            # NORMALIZATION
            # ------------------------------------------------

            chunk = normalize_dataframe(chunk)

            # ------------------------------------------------
            # EXTRA BLOCKING COLUMNS
            # ------------------------------------------------

            chunk["name_prefix4"] = (
                chunk["name_norm"]
                .fillna("")
                .str[:4]
            )

            chunk["name_prefix6"] = (
                chunk["name_norm"]
                .fillna("")
                .str[:6]
            )

            # ------------------------------------------------
            # PYARROW
            # ------------------------------------------------

            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False,
            )

            # First chunk creates the Parquet schema.
            if parquet_writer is None:

                parquet_writer = pq.ParquetWriter(
                    local_path,
                    table.schema,
                    compression="snappy",
                )

            parquet_writer.write_table(table)

            total_rows += len(chunk)

            print(
                f"  Total processed: {total_rows:,}"
            )

            # Explicitly release the chunk.
            del chunk
            del table

    finally:

        if parquet_writer is not None:
            parquet_writer.close()

        body.close()

    print()
    print(f"Finished processing {total_rows:,} rows.")
    print(f"Local Parquet: {local_path}")

    # --------------------------------------------------------
    # UPLOAD TO S3
    # --------------------------------------------------------

    print()
    print("Uploading Parquet to S3...")

    s3.upload_file(
        local_path,
        output_bucket,
        output_key,
    )

    print()
    print("Upload complete.")
    print(f"s3://{output_bucket}/{output_key}")

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    try:
        os.remove(local_path)
    except OSError:
        pass

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Rows processed: {total_rows:,}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Memory-safe full dataset preprocessing"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input S3 TSV path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output S3 Parquet path",
    )

    args = parser.parse_args()

    process_file(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
