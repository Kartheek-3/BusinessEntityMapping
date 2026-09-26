import argparse
import os
import sys

import boto3

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from src.config import (
    AWS_REGION,
    BUCKET,
    RAW_PREFIX,
    PROCESSED_PREFIX,
)


def main():
    parser = argparse.ArgumentParser(
        description="Launch SageMaker preprocessing job"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["train", "test"],
    )

    parser.add_argument(
        "--source-name",
        required=True,
        choices=["source1", "source2", "source3"],
    )

    parser.add_argument(
        "--role",
        required=True,
        help="SageMaker execution role ARN",
    )

    parser.add_argument(
        "--instance-type",
        default="ml.m5.xlarge",
    )

    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--input-s3",
        default=None,
    )

    parser.add_argument(
        "--output-s3",
        default=None,
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required before launching an AWS job",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Confirmation gate
    # ---------------------------------------------------------

    if not args.confirm:
        raise SystemExit(
            "AWS job not launched. Add --confirm to continue."
        )

    # ---------------------------------------------------------
    # S3 paths
    # ---------------------------------------------------------

    input_s3 = (
        args.input_s3
        or (
            f"s3://{BUCKET}/{RAW_PREFIX}/"
            f"{args.dataset}/"
            f"{args.dataset}_{args.source_name}.tsv"
        )
    )

    output_s3 = (
        args.output_s3
        or (
            f"s3://{BUCKET}/{PROCESSED_PREFIX}/"
            f"{args.dataset}/"
        )
    )

    input_filename = os.path.basename(input_s3)

    output_filename = (
        f"{args.dataset}_{args.source_name}.parquet"
    )

    print("=" * 70)
    print("SageMaker Preprocessing Job")
    print("=" * 70)
    print(f"Region:          {AWS_REGION}")
    print(f"Dataset:         {args.dataset}")
    print(f"Source:          {args.source_name}")
    print(f"Input:           {input_s3}")
    print(f"Output:          {output_s3}")
    print(f"Output file:     {output_filename}")
    print(f"Instance:        {args.instance_type}")
    print(f"Instances:       {args.instance_count}")
    print("=" * 70)

    # ---------------------------------------------------------
    # AWS session
    # ---------------------------------------------------------

    boto3_session = boto3.Session(
        region_name=AWS_REGION
    )

    sagemaker_session = Session(
        boto_session=boto3_session
    )

    # ---------------------------------------------------------
    # Scikit-learn Processing image
    # ---------------------------------------------------------

    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=AWS_REGION,
        version="1.2-1",
        py_version="py3",
        instance_type=args.instance_type,
    )

    print(f"Processing image: {sklearn_image}")

    # ---------------------------------------------------------
    # ScriptProcessor
    # ---------------------------------------------------------

    processor = ScriptProcessor(
        image_uri=sklearn_image,
        role=args.role,
        command=["python3"],
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        base_job_name=(
            f"business-er-preprocess-"
            f"{args.dataset}-"
            f"{args.source_name}"
        ),
        sagemaker_session=sagemaker_session,
    )

    # ---------------------------------------------------------
    # Processing input
    # ---------------------------------------------------------

    processing_input = ProcessingInput(
        input_name="input_data",
        s3_input=ProcessingS3Input(
            s3_uri=input_s3,
            s3_data_type="S3Prefix",
            local_path="/opt/ml/processing/input/data",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
        ),
    )

    # ---------------------------------------------------------
    # Processing output
    # ---------------------------------------------------------

    processing_output = ProcessingOutput(
        output_name="output_data",
        s3_output=ProcessingS3Output(
            s3_uri=output_s3,
            s3_upload_mode="EndOfJob",
            local_path="/opt/ml/processing/output/data",
        ),
    )

    # ---------------------------------------------------------
    # Launch Processing job
    # ---------------------------------------------------------

    print("Submitting SageMaker Processing job...")

    processor.run(
        code="jobs/preprocessing_job.py",

        inputs=[
            processing_input,
        ],

        outputs=[
            processing_output,
        ],

        arguments=[
            "--input",
            f"/opt/ml/processing/input/data/{input_filename}",
            "--output",
            (
                f"/opt/ml/processing/output/data/"
                f"{output_filename}"
            ),
        ],

        wait=True,
        logs=True,
    )

    print()
    print("=" * 70)
    print("PREPROCESSING JOB COMPLETED")
    print("=" * 70)
    print(
        f"Output: {output_s3}{output_filename}"
    )


if __name__ == "__main__":
    main()