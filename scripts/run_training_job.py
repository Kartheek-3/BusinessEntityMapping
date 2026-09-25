import argparse
import os
import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import AWS_REGION, BUCKET, FEATURE_PREFIX, MODEL_PREFIX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--confirm", action="store_true", help="Execute the job on AWS")
    
    args = parser.parse_args()

    train_s3 = f"s3://{BUCKET}/{FEATURE_PREFIX}/train/features.parquet"
    output_s3 = f"s3://{BUCKET}/{MODEL_PREFIX}/"
    
    print(f"AWS Region: {AWS_REGION}")
    print(f"Bucket: {BUCKET}")
    print(f"Train Input: {train_s3}")
    print(f"Model Output: {output_s3}")
    
    if not args.confirm:
        print("Dry run complete. Use --confirm to submit to SageMaker.")
        return

    boto3_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)

    # Note: training_job.py assumes input is a parquet file via args.train, but SageMaker XGBoost estimator 
    # natively supports parquet. However, since we defined our own entry point for custom split validation,
    # we'll use XGBoost as a script mode estimator.
    
    estimator = XGBoost(
        entry_point="jobs/training_job.py",
        dependencies=["src/"],
        framework_version="1.7-1",
        py_version="py3",
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=output_s3,
        sagemaker_session=sagemaker_session,
        base_job_name="business-er-training",
        hyperparameters={
            "train": "/opt/ml/input/data/train/features.parquet",
            "model-dir": "/opt/ml/model"
        }
    )

    train_input = TrainingInput(
        s3_data=train_s3,
        content_type="application/x-parquet"
    )

    estimator.fit({"train": train_input})
    print("Training job completed.")

if __name__ == "__main__":
    main()
