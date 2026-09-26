import argparse
import os
import sys

import boto3
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

BUCKET = "kartheek-business-er-2026"
REGION = "eu-north-1"

S3_PROCESSED_PREFIX = "processed"

DEFAULT_DATABASE = "models/blocking.duckdb"
DEFAULT_LOCAL_DIR = "data/local"


# ============================================================
# PATH HELPERS
# ============================================================

def s3_uri(dataset, source):
    return (
        f"s3://{BUCKET}/{S3_PROCESSED_PREFIX}/"
        f"{dataset}/{source}.parquet"
    )


def local_path(local_dir, dataset, source):
    return os.path.join(
        local_dir,
        dataset,
        f"{source}.parquet"
    )


# ============================================================
# DOWNLOAD FROM S3
# ============================================================

def download_if_missing(
    s3_client,
    dataset,
    source,
    local_dir
):
    """
    Download one processed Parquet file from S3
    if it does not already exist locally.
    """

    s3_key = (
        f"{S3_PROCESSED_PREFIX}/"
        f"{dataset}/"
        f"{source}.parquet"
    )

    output_path = local_path(
        local_dir,
        dataset,
        source
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)

        print(
            f"[LOCAL] {source}: already exists "
            f"({size_mb:.1f} MB)"
        )

        return output_path

    print()
    print("=" * 70)
    print(f"Downloading {source}")
    print("=" * 70)

    print(f"S3:     s3://{BUCKET}/{s3_key}")
    print(f"Local:  {output_path}")

    s3_client.download_file(
        BUCKET,
        s3_key,
        output_path
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(
        f"Downloaded successfully: "
        f"{size_mb:.1f} MB"
    )

    return output_path


# ============================================================
# DUCKDB CONFIGURATION
# ============================================================

def configure_duckdb(con):
    """
    Configure DuckDB for the limited-memory Code Editor machine.
    """

    # Keep RAM usage controlled.
    con.execute(
        "SET memory_limit='2GB'"
    )

    # Use only two threads because the Code Editor
    # has limited CPU/RAM.
    con.execute(
        "SET threads=2"
    )

    # Reduces unnecessary memory usage.
    con.execute(
        "SET preserve_insertion_order=false"
    )

    # Allow DuckDB to spill temporary data to disk.
    temp_dir = "models/duckdb_tmp"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    con.execute(
        f"SET temp_directory='{temp_dir}'"
    )

    # These settings are retained in case DuckDB
    # accesses S3 for any metadata/files.
    con.execute(
        "SET http_timeout=300"
    )

    con.execute(
        "SET http_retries=10"
    )

    con.execute(
        "SET http_retry_wait_ms=1000"
    )


# ============================================================
# CREATE LOCAL DUCKDB TABLES
# ============================================================

def create_local_tables(
    con,
    source_paths
):
    """
    Materialize the required columns from local Parquet
    into DuckDB tables.

    This means the large candidate joins no longer
    repeatedly read files from S3.
    """

    print()
    print("=" * 70)
    print("CREATING LOCAL DUCKDB TABLES")
    print("=" * 70)

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    print("\nLoading Source 1...")

    con.execute("DROP TABLE IF EXISTS s1")

    con.execute(
        f"""
        CREATE TABLE s1 AS
        SELECT
            entity_id,
            name_norm,
            country_norm,
            name_prefix4,
            name_prefix6
        FROM read_parquet(
            '{source_paths["source1"]}'
        )
        """
    )

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    print("Loading Source 2...")

    con.execute("DROP TABLE IF EXISTS s2")

    con.execute(
        f"""
        CREATE TABLE s2 AS
        SELECT
            entity_id,
            name_norm,
            country_norm,
            name_prefix4,
            name_prefix6
        FROM read_parquet(
            '{source_paths["source2"]}'
        )
        """
    )

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    print("Loading Source 3...")

    con.execute("DROP TABLE IF EXISTS s3")

    con.execute(
        f"""
        CREATE TABLE s3 AS
        SELECT
            entity_id,
            name_norm,
            country_norm,
            name_prefix4,
            name_prefix6
        FROM read_parquet(
            '{source_paths["source3"]}'
        )
        """
    )

    print("\nLocal DuckDB tables created.")


# ============================================================
# ROW COUNTS
# ============================================================

def print_row_counts(con):

    print()
    print("=" * 70)
    print("ROW COUNTS")
    print("=" * 70)

    for source in ["s1", "s2", "s3"]:

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {source}
            """
        ).fetchone()[0]

        print(
            f"{source.upper()} rows: "
            f"{count:,}"
        )


# ============================================================
# BLOCKING KEY STATISTICS
# ============================================================

def print_blocking_key_stats(con):

    print()
    print("=" * 70)
    print("BLOCKING KEY STATISTICS")
    print("=" * 70)

    for source in ["s2", "s3"]:

        print()
        print(f"{source.upper()} PREFIX-4")

        result = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_keys,
                AVG(cnt) AS avg_frequency,
                MAX(cnt) AS max_frequency,
                SUM(
                    CASE
                        WHEN cnt <= 500
                        THEN 1
                        ELSE 0
                    END
                ) AS eligible_keys
            FROM (
                SELECT
                    country_norm,
                    name_prefix4,
                    COUNT(*) AS cnt
                FROM {source}
                WHERE
                    name_prefix4 <> ''
                GROUP BY
                    country_norm,
                    name_prefix4
            )
            """
        ).fetchone()

        print(
            f"Total keys:      {result[0]:,}"
        )
        print(
            f"Average freq:    {result[1]:.2f}"
        )
        print(
            f"Max freq:        {result[2]:,}"
        )
        print(
            f"Keys <= 500:     {result[3]:,}"
        )

        print()
        print(f"{source.upper()} PREFIX-6")

        result = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_keys,
                AVG(cnt) AS avg_frequency,
                MAX(cnt) AS max_frequency,
                SUM(
                    CASE
                        WHEN cnt <= 1000
                        THEN 1
                        ELSE 0
                    END
                ) AS eligible_keys
            FROM (
                SELECT
                    country_norm,
                    name_prefix6,
                    COUNT(*) AS cnt
                FROM {source}
                WHERE
                    name_prefix6 <> ''
                GROUP BY
                    country_norm,
                    name_prefix6
            )
            """
        ).fetchone()

        print(
            f"Total keys:      {result[0]:,}"
        )
        print(
            f"Average freq:    {result[1]:.2f}"
        )
        print(
            f"Max freq:        {result[2]:,}"
        )
        print(
            f"Keys <= 1000:    {result[3]:,}"
        )


# ============================================================
# CREATE ELIGIBLE BLOCKING KEYS
# ============================================================

def create_blocking_keys(con):

    print()
    print("=" * 70)
    print("CREATING BLOCKING KEY TABLES")
    print("=" * 70)

    # --------------------------------------------------------
    # S2 PREFIX 4
    # --------------------------------------------------------

    print("\nS2 prefix-4 keys...")

    con.execute(
        """
        DROP TABLE IF EXISTS s2_prefix4_keys
        """
    )

    con.execute(
        """
        CREATE TABLE s2_prefix4_keys AS
        SELECT
            country_norm,
            name_prefix4
        FROM s2
        WHERE
            name_prefix4 <> ''
        GROUP BY
            country_norm,
            name_prefix4
        HAVING
            COUNT(*) <= 500
        """
    )

    # --------------------------------------------------------
    # S3 PREFIX 4
    # --------------------------------------------------------

    print("S3 prefix-4 keys...")

    con.execute(
        """
        DROP TABLE IF EXISTS s3_prefix4_keys
        """
    )

    con.execute(
        """
        CREATE TABLE s3_prefix4_keys AS
        SELECT
            country_norm,
            name_prefix4
        FROM s3
        WHERE
            name_prefix4 <> ''
        GROUP BY
            country_norm,
            name_prefix4
        HAVING
            COUNT(*) <= 500
        """
    )

    # --------------------------------------------------------
    # S2 PREFIX 6
    # --------------------------------------------------------

    print("S2 prefix-6 keys...")

    con.execute(
        """
        DROP TABLE IF EXISTS s2_prefix6_keys
        """
    )

    con.execute(
        """
        CREATE TABLE s2_prefix6_keys AS
        SELECT
            country_norm,
            name_prefix6
        FROM s2
        WHERE
            name_prefix6 <> ''
        GROUP BY
            country_norm,
            name_prefix6
        HAVING
            COUNT(*) <= 1000
        """
    )

    # --------------------------------------------------------
    # S3 PREFIX 6
    # --------------------------------------------------------

    print("S3 prefix-6 keys...")

    con.execute(
        """
        DROP TABLE IF EXISTS s3_prefix6_keys
        """
    )

    con.execute(
        """
        CREATE TABLE s3_prefix6_keys AS
        SELECT
            country_norm,
            name_prefix6
        FROM s3
        WHERE
            name_prefix6 <> ''
        GROUP BY
            country_norm,
            name_prefix6
        HAVING
            COUNT(*) <= 1000
        """
    )

    print("\nBlocking key tables created.")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(con):

    print()
    print("=" * 70)
    print("GENERATING CANDIDATES")
    print("=" * 70)

    # --------------------------------------------------------
    # Start with an empty candidate table.
    #
    # PRIMARY KEY prevents duplicate pairs from being stored.
    # --------------------------------------------------------

    con.execute(
        """
        DROP TABLE IF EXISTS candidate_pairs_long
        """
    )

    con.execute(
        """
        CREATE TABLE candidate_pairs_long (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
        """
    )

    # ========================================================
    # RULE 1
    # EXACT NORMALIZED NAME
    # ========================================================

    print("\n[1/6] Exact normalized name - S2")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s2.entity_id
        FROM s1
        JOIN s2
          ON s1.name_norm <> ''
         AND s1.name_norm = s2.name_norm
        """
    )

    print("[2/6] Exact normalized name - S3")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s3.entity_id
        FROM s1
        JOIN s3
          ON s1.name_norm <> ''
         AND s1.name_norm = s3.name_norm
        """
    )

    # ========================================================
    # RULE 2
    # COUNTRY + PREFIX 4
    # ========================================================

    print("[3/6] Country + prefix4 - S2")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s2.entity_id
        FROM s1
        JOIN s2
          ON s1.country_norm = s2.country_norm
         AND s1.name_prefix4 = s2.name_prefix4
         AND s1.name_prefix4 <> ''
        JOIN s2_prefix4_keys k
          ON k.country_norm = s2.country_norm
         AND k.name_prefix4 = s2.name_prefix4
        """
    )

    print("[4/6] Country + prefix4 - S3")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s3.entity_id
        FROM s1
        JOIN s3
          ON s1.country_norm = s3.country_norm
         AND s1.name_prefix4 = s3.name_prefix4
         AND s1.name_prefix4 <> ''
        JOIN s3_prefix4_keys k
          ON k.country_norm = s3.country_norm
         AND k.name_prefix4 = s3.name_prefix4
        """
    )

    # ========================================================
    # RULE 3
    # COUNTRY + PREFIX 6
    # ========================================================

    print("[5/6] Country + prefix6 - S2")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s2.entity_id
        FROM s1
        JOIN s2
          ON s1.country_norm = s2.country_norm
         AND s1.name_prefix6 = s2.name_prefix6
         AND s1.name_prefix6 <> ''
        JOIN s2_prefix6_keys k
          ON k.country_norm = s2.country_norm
         AND k.name_prefix6 = s2.name_prefix6
        """
    )

    print("[6/6] Country + prefix6 - S3")

    con.execute(
        """
        INSERT OR IGNORE INTO candidate_pairs_long
        SELECT
            s1.entity_id,
            s3.entity_id
        FROM s1
        JOIN s3
          ON s1.country_norm = s3.country_norm
         AND s1.name_prefix6 = s3.name_prefix6
         AND s1.name_prefix6 <> ''
        JOIN s3_prefix6_keys k
          ON k.country_norm = s3.country_norm
         AND k.name_prefix6 = s3.name_prefix6
        """
    )

    print("\nCandidate generation completed.")


# ============================================================
# CANDIDATE STATISTICS
# ============================================================

def print_candidate_statistics(con):

    print()
    print("=" * 70)
    print("CANDIDATE STATISTICS")
    print("=" * 70)

    total_candidates = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs_long
        """
    ).fetchone()[0]

    print(
        f"Total candidate pairs: "
        f"{total_candidates:,}"
    )

    unique_s1 = con.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM candidate_pairs_long
        """
    ).fetchone()[0]

    total_s1 = con.execute(
        """
        SELECT COUNT(*)
        FROM s1
        """
    ).fetchone()[0]

    coverage = (
        unique_s1 / total_s1 * 100
        if total_s1
        else 0
    )

    print(
        f"S1 entities with candidates: "
        f"{unique_s1:,} / {total_s1:,}"
    )

    print(
        f"Candidate coverage: "
        f"{coverage:.2f}%"
    )

    # --------------------------------------------------------
    # Candidate count distribution
    # --------------------------------------------------------

    stats = con.execute(
        """
        SELECT
            AVG(candidate_count),
            quantile_cont(candidate_count, 0.50),
            quantile_cont(candidate_count, 0.95),
            quantile_cont(candidate_count, 0.99),
            MAX(candidate_count)
        FROM (
            SELECT
                source1_entity_id,
                COUNT(*) AS candidate_count
            FROM candidate_pairs_long
            GROUP BY source1_entity_id
        )
        """
    ).fetchone()

    avg_candidates = stats[0] or 0
    median_candidates = stats[1] or 0
    p95 = stats[2] or 0
    p99 = stats[3] or 0
    max_candidates = stats[4] or 0

    print(
        f"Average candidates/S1: "
        f"{avg_candidates:.2f}"
    )

    print(
        f"Median candidates/S1: "
        f"{median_candidates:.0f}"
    )

    print(
        f"P95 candidates/S1: "
        f"{p95:.0f}"
    )

    print(
        f"P99 candidates/S1: "
        f"{p99:.0f}"
    )

    print(
        f"Max candidates/S1: "
        f"{max_candidates:,}"
    )


# ============================================================
# WRITE FINAL CANDIDATE FILE
# ============================================================

def write_candidate_file(
    con,
    output_path
):

    print()
    print("=" * 70)
    print("WRITING candidate_pairs.tsv")
    print("=" * 70)

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # DuckDB writes directly to the output file.
    con.execute(
        f"""
        COPY (
            SELECT
                s1.entity_id AS source1_entity_id,

                COALESCE(
                    STRING_AGG(
                        c.candidate_entity_id,
                        ','
                        ORDER BY
                            c.candidate_entity_id
                    ),
                    ''
                ) AS candidate_entity_ids

            FROM s1

            LEFT JOIN candidate_pairs_long c
              ON s1.entity_id =
                 c.source1_entity_id

            GROUP BY
                s1.entity_id

            ORDER BY
                s1.entity_id
        )
        TO '{output_path}'
        (
            HEADER,
            DELIMITER '\\t'
        )
        """
    )

    print(
        f"\nCandidate file written:"
        f"\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "DuckDB-based business entity "
            "resolution blocking"
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "train",
            "test"
        ],
        required=True,
        help="Dataset to process"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output candidate_pairs.tsv path"
    )

    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="DuckDB database path"
    )

    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=(
            "Local directory used to cache "
            "processed Parquet files"
        )
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "Do not download files; "
            "require local Parquet files to exist"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(args.database) or ".",
        exist_ok=True
    )

    os.makedirs(
        args.local_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DUCKDB BLOCKING")
    print("=" * 70)

    print(
        f"Dataset:       {args.dataset}"
    )

    print(
        f"Local cache:   {args.local_dir}"
    )

    print(
        f"Database:      {args.database}"
    )

    print(
        f"Output:        {args.output}"
    )

    # --------------------------------------------------------
    # S3 client
    # --------------------------------------------------------

    s3_client = boto3.client(
        "s3",
        region_name=REGION
    )

    # --------------------------------------------------------
    # Download processed Parquet files
    # --------------------------------------------------------

    source_paths = {}

    for source in [
        "source1",
        "source2",
        "source3"
    ]:

        if args.skip_download:

            path = local_path(
                args.local_dir,
                args.dataset,
                source
            )

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"Required local file does not exist: "
                    f"{path}"
                )

            print(
                f"[LOCAL] Using {path}"
            )

            source_paths[source] = path

        else:

            source_paths[source] = (
                download_if_missing(
                    s3_client=s3_client,
                    dataset=args.dataset,
                    source=source,
                    local_dir=args.local_dir
                )
            )

    # --------------------------------------------------------
    # Connect to DuckDB
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONNECTING TO DUCKDB")
    print("=" * 70)

    con = duckdb.connect(
        args.database
    )

    try:

        configure_duckdb(con)

        # ----------------------------------------------------
        # Materialize local source tables
        # ----------------------------------------------------

        create_local_tables(
            con,
            source_paths
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        print_row_counts(con)

        # ----------------------------------------------------
        # Blocking key statistics
        # ----------------------------------------------------

        print_blocking_key_stats(con)

        # ----------------------------------------------------
        # Create eligible blocking keys
        # ----------------------------------------------------

        create_blocking_keys(con)

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        generate_candidates(con)

        # ----------------------------------------------------
        # Candidate statistics
        # ----------------------------------------------------

        print_candidate_statistics(con)

        # ----------------------------------------------------
        # Write candidate file
        # ----------------------------------------------------

        write_candidate_file(
            con,
            args.output
        )

        print()
        print("=" * 70)
        print("BLOCKING COMPLETE")
        print("=" * 70)

    finally:

        con.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()