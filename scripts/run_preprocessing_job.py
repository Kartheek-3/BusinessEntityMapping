import argparse
import os
import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Add current directory to path if running locally to find src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AWS_REGION, BUCKET, RAW_PREFIX, PROCESSED_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-name", required=True, help="e.g., source1, source2, source3")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--input-s3", help="Override default S3 input path (must be file)")
    parser.add_argument("--output-s3", help="Override default S3 output path (must be directory)")
    
    args = parser.parse_args()

    # Determine paths
    # Note: input_s3 is expected to point to a specific file like train_source1.tsv
    input_s3 = args.input_s3 or f"s3://{BUCKET}/{RAW_PREFIX}/train/train_{args.source_name}.tsv"
    # Output S3 is a prefix where the output parquet will be written
    output_s3 = args.output_s3 or f"s3://{BUCKET}/{PROCESSED_PREFIX}/train"
    
    input_filename = os.path.basename(input_s3)
    output_filename = f"train_{args.source_name}.parquet"
    
    print(f"AWS Region: {AWS_REGION}")
    print(f"Bucket: {BUCKET}")
    print(f"Source: {args.source_name}")
    print(f"Input S3: {input_s3}")
    print(f"Output S3: {output_s3}")

    boto3_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        sagemaker_session=sagemaker_session,
        base_job_name=f"business-er-preprocess-{args.source_name}"
    )

    processor.run(
        code="jobs/preprocessing_job.py",
        inputs=[
            ProcessingInput(
                source=input_s3,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                source="src/",
                destination="/opt/ml/processing/input/code/src",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/data",
                destination=output_s3,
            )
        ],
        arguments=[
            "--input", f"/opt/ml/processing/input/data/{input_filename}",
            "--output", f"/opt/ml/processing/output/data/{output_filename}"
        ],
        wait=True
    )
    
    print(f"Preprocessing job for {args.source_name} completed successfully.")

if __name__ == "__main__":
    main()
