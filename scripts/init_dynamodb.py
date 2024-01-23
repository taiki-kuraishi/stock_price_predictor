"""
This module provides functions to initialize and manage DynamoDB tables.

It includes functions to initialize the limit table, initialize the stock table,
delete the prediction table, and initialize all models.
"""
import os
from tqdm import tqdm
import boto3
import pandas as pd
from dotenv import load_dotenv
from mypy_boto3_dynamodb import DynamoDBServiceResource
from mypy_boto3_dynamodb.service_resource import Table
from mypy_boto3_s3 import S3ServiceResource


def init_limit_table(
    table: Table,
) -> None:
    """
    Initialize the limit table.

    This function scans the table and deletes any items it finds.

    Args:
        table (Table): The table to initialize.

    Returns:
        None
    """
    to_delete_df = pd.DataFrame(
        table.scan()["Items"],
    )

    if not to_delete_df.empty:
        only_key_to_delete_df = to_delete_df[["stock_id", "operation"]]
        to_delete_list: list[dict[str, str]] = [
            {str(k): str(v) for k, v in d.items()} for d in only_key_to_delete_df.to_dict("records")
        ]
        for key in tqdm(to_delete_list):
            # print(key)
            table.delete_item(Key=key)

    return None


if __name__ == "__main__":
    load_dotenv(dotenv_path="../ml_lambda/.env.ml", override=True, verbose=True)

    aws_config: dict[str, str] = {
        "region_name": os.environ["REGION_NAME"],
        "access_key_id": os.environ["ACCESS_KEY_ID"],
        "secret_access_key": os.environ["SECRET_ACCESS_KEY"],
    }
    dynamodb_tables: dict[str, str] = {
        "stock": os.environ["AWS_DYNAMODB_STOCK_TABLE_NAME"],
        "prediction": os.environ["AWS_DYNAMODB_PREDICTION_TABLE_NAME"],
        "limit": os.environ["AWS_DYNAMODB_LIMIT_TABLE_NAME"],
    }
    s3_bucket = os.environ["AWS_S3_BUCKET_NAME"]

    # aws instance
    dynamodb: DynamoDBServiceResource = boto3.resource(
        "dynamodb",
        region_name=aws_config["region_name"],
        aws_access_key_id=aws_config["access_key_id"],
        aws_secret_access_key=aws_config["secret_access_key"],
    )
    s3: S3ServiceResource = boto3.resource(
        "s3",
        aws_access_key_id=aws_config["access_key_id"],
        aws_secret_access_key=aws_config["secret_access_key"],
    )
    stock_table: Table = dynamodb.Table(dynamodb_tables["stock"])
    prediction_table: Table = dynamodb.Table(dynamodb_tables["prediction"])
    limit_table: Table = dynamodb.Table(dynamodb_tables["limit"])

    if "y" == input("Do you want to init limit table? [y/n]"):
        init_limit_table(limit_table)
