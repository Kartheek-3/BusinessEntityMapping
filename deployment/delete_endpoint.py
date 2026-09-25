import boto3


REGION = "eu-north-1"

ENDPOINT_NAME = (
    "business-entity-resolution"
)


def main():

    client = boto3.client(
        "sagemaker",
        region_name=REGION,
    )

    client.delete_endpoint(
        EndpointName=ENDPOINT_NAME
    )

    print(
        f"Deleted {ENDPOINT_NAME}"
    )


if __name__ == "__main__":
    main()
