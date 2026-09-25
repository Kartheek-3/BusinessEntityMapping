import boto3
import sagemaker

from sagemaker.model import Model
from sagemaker.image_uris import retrieve


REGION = "eu-north-1"

BUCKET = "kartheek-business-er-2026"

MODEL_DATA = (
    f"s3://{BUCKET}/models/xgboost/model.tar.gz"
)

ENDPOINT_NAME = (
    "business-entity-resolution"
)


def main():

    session = sagemaker.Session()

    role = sagemaker.get_execution_role()

    image_uri = retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1",
        py_version="py3",
        instance_type="ml.m5.large",
        image_scope="inference",
    )

    model = Model(
        image_uri=image_uri,
        model_data=MODEL_DATA,
        role=role,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )

    print(
        "Endpoint:",
        predictor.endpoint_name,
    )


if __name__ == "__main__":
    main()
