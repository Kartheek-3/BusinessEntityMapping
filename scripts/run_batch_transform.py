import argparse
import os
import boto3
import sagemaker
from sagemaker.model import Model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import AWS_REGION, BUCKET, FEATURE_PREFIX, PREDICTION_PREFIX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--model-s3", required=True, help="S3 URI to the model.tar.gz artifact")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--confirm", action="store_true", help="Execute the job on AWS")
    
    args = parser.parse_args()

    test_features_s3 = f"s3://{BUCKET}/{FEATURE_PREFIX}/test/"
    output_s3 = f"s3://{BUCKET}/{PREDICTION_PREFIX}/test/"
    
    print(f"AWS Region: {AWS_REGION}")
    print(f"Bucket: {BUCKET}")
    print(f"Model: {args.model_s3}")
    print(f"Test Input: {test_features_s3}")
    print(f"Output: {output_s3}")
    
    if not args.confirm:
        print("Dry run complete. Use --confirm to submit to SageMaker.")
        return

    boto3_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)

    # For Batch Transform with XGBoost
    from sagemaker.image_uris import retrieve
    image_uri = retrieve("xgboost", AWS_REGION, "1.7-1")

    model = Model(
        image_uri=image_uri,
        model_data=args.model_s3,
        role=args.role,
        sagemaker_session=sagemaker_session
    )

    transformer = model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=output_s3,
        accept="application/x-parquet",
        assemble_with="Line"
    )

    transformer.transform(
        data=test_features_s3,
        content_type="application/x-parquet",
        split_type="Line"
    )
    
    transformer.wait()
    print("Batch transform completed.")

if __name__ == "__main__":
    main()
