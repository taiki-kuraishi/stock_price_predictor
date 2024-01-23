"""
This module provides functions to initialize and manage DynamoDB tables.

It includes functions to initialize the limit table, initialize the stock table,
delete the prediction table, and initialize all models.
"""
import os
from datetime import datetime, tzinfo
from decimal import Decimal

import boto3
import pandas as pd
import pytz
from dotenv import load_dotenv
from mypy_boto3_dynamodb import DynamoDBServiceResource
from mypy_boto3_dynamodb.service_resource import Table
from mypy_boto3_s3 import S3ServiceResource
from mypy_boto3_s3.service_resource import Bucket
from tqdm import tqdm


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
            {str(k): str(v) for k, v in d.items()}
            for d in only_key_to_delete_df.to_dict("records")
        ]
        for key in tqdm(to_delete_list):
            # print(key)
            table.delete_item(Key=key)

    return None


def init_stock_table(
    stock_table: Table,
    limit_table: Table,
    bucket: Bucket,
    timezone: tzinfo,
    tmp_dir: str,
    target_stock: str,
    stock_name: str,
    period: str,
    interval: str,
) -> None:
    """
    Initialize the stock table.

    This function scans the table and deletes any items it finds.

    Args:
        table (Table): The table to initialize.

    Returns:
        None
    """
    # scan
    to_delete_df = pd.DataFrame(
        stock_table.scan()["Items"],
    )

    # delete all items
    if not to_delete_df.empty:
        only_key_to_delete_df = to_delete_df[["date", "time"]]
        to_delete_list: list[dict[str, str]] = [
            {str(k): str(v) for k, v in d.items()}
            for d in only_key_to_delete_df.to_dict("records")
        ]
        for key in tqdm(to_delete_list):
            # print(key)
            stock_table.delete_item(Key=key)

    # download original csv from s3
    file_name = f"spp_{stock_name}_{period}_{interval}.csv"
    bucket.download_file(f"csv/{file_name}", f"{tmp_dir}/{file_name}")

    # csv to DataFrame
    original_df: pd.DataFrame = pd.read_csv(
        f"{tmp_dir}/{file_name}", encoding="utf-8", index_col=None
    )

    # convert float to Decimal
    original_df = original_df.apply(
        lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
    )

    # upload to dynamodb stock table
    with stock_table.batch_writer() as batch:
        for item in tqdm(original_df.to_dict("records")):
            batch.put_item(Item=item)

    # update limit table
    limit_table.put_item(
        Item={
            "stock_id": target_stock,
            "operation": "stock",
            "create_at": datetime.now(timezone).isoformat(),
            "max": pd.to_datetime(
                original_df.tail(1)["date"].values[0]
                + " "
                + original_df.tail(1)["time"].values[0]
            ).isoformat(),
        }
    )
    return None


if __name__ == "__main__":
    load_dotenv(dotenv_path="../ml_lambda/.env.ml", override=True, verbose=True)

    lambda_config: dict[str | tzinfo, str] = {
        "timezone": pytz.timezone(os.environ["TIMEZONE"]),
        "tmp_dir": os.environ["TMP_DIR"],
    }

    stock_config: dict[str, str] = {
        "target_stock": os.environ["TARGET_STOCK"],
        "stock_name": os.environ["STOCK_NAME"],
        "period": os.environ["PERIOD"],
        "interval": os.environ["INTERVAL"],
    }

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
    s3_bucket_name = os.environ["AWS_S3_BUCKET_NAME"]

    # aws instance
    dynamodb: DynamoDBServiceResource = boto3.resource(
        "dynamodb",
        region_name=aws_config["region_name"],
        aws_access_key_id=aws_config["access_key_id"],
        aws_secret_access_key=aws_config["secret_access_key"],
    )
    stock_table_instance: Table = dynamodb.Table(dynamodb_tables["stock"])
    prediction_table_instance: Table = dynamodb.Table(dynamodb_tables["prediction"])
    limit_table_instance: Table = dynamodb.Table(dynamodb_tables["limit"])

    s3: S3ServiceResource = boto3.resource(
        "s3",
        aws_access_key_id=aws_config["access_key_id"],
        aws_secret_access_key=aws_config["secret_access_key"],
    )
    bucket_instance: Bucket = s3.Bucket(s3_bucket_name)

    if "y" == input("Do you want to init limit table? [y/n]"):
        init_limit_table(bucket_instance)

    if "y" == input("Do you want to init stock table? [y/n]"):
        init_stock_table(
            stock_table_instance,
            limit_table_instance,
            bucket_instance,
            **lambda_config,
            **stock_config,
        )
