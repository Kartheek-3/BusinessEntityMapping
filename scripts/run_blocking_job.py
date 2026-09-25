import argparse
import os
import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import AWS_REGION, BUCKET, PROCESSED_PREFIX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--dataset", choices=["train", "test"], required=True)
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for quick validation")
    parser.add_argument("--confirm", action="store_true", help="Execute the job on AWS")
    
    args = parser.parse_args()

    s1_s3 = f"s3://{BUCKET}/{PROCESSED_PREFIX}/{args.dataset}/train_source1.parquet" if args.dataset == "train" else f"s3://{BUCKET}/{PROCESSED_PREFIX}/test/test_source1.parquet"
    s2_s3 = f"s3://{BUCKET}/{PROCESSED_PREFIX}/{args.dataset}/train_source2.parquet" if args.dataset == "train" else f"s3://{BUCKET}/{PROCESSED_PREFIX}/test/test_source2.parquet"
    s3_s3 = f"s3://{BUCKET}/{PROCESSED_PREFIX}/{args.dataset}/train_source3.parquet" if args.dataset == "train" else f"s3://{BUCKET}/{PROCESSED_PREFIX}/test/test_source3.parquet"
    output_s3 = f"s3://{BUCKET}/candidates/{args.dataset}/"
    
    print(f"AWS Region: {AWS_REGION}")
    print(f"Bucket: {BUCKET}")
    print(f"S1 Input: {s1_s3}")
    print(f"S2 Input: {s2_s3}")
    print(f"S3 Input: {s3_s3}")
    print(f"Output: {output_s3}candidate_pairs.tsv")
    
    if not args.confirm:
        print("Dry run complete. Use --confirm to submit to SageMaker.")
        return

    boto3_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        sagemaker_session=sagemaker_session,
        base_job_name=f"business-er-blocking-{args.dataset}"
    )

    processor.run(
        code="jobs/blocking_job.py",
        inputs=[
            ProcessingInput(source=s1_s3, destination="/opt/ml/processing/input/s1"),
            ProcessingInput(source=s2_s3, destination="/opt/ml/processing/input/s2"),
            ProcessingInput(source=s3_s3, destination="/opt/ml/processing/input/s3"),
            ProcessingInput(source="src/", destination="/opt/ml/processing/input/code/src")
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output/data", destination=output_s3)
        ],
        arguments=[
            "--s1", "/opt/ml/processing/input/s1/" + os.path.basename(s1_s3),
            "--s2", "/opt/ml/processing/input/s2/" + os.path.basename(s2_s3),
            "--s3", "/opt/ml/processing/input/s3/" + os.path.basename(s3_s3),
            "--output", "/opt/ml/processing/output/data/candidate_pairs.tsv"
        ] + (["--subset"] if args.subset else []),
        wait=True
    )
    print("Blocking job completed.")

if __name__ == "__main__":
    main()
