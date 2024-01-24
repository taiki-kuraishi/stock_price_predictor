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
from dataclass.configurations import (
    AWSAuth,
    DynamoDBTables,
    LambdaConfiguration,
    StockConfiguration,
)
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
            table.delete_item(Key=key)


def init_stock_table(
    stock_table: Table,
    limit_table: Table,
    bucket: Bucket,
    lambda_config: LambdaConfiguration,
    stock_config: StockConfiguration,
) -> None:
    """
    Initialize the stock table.

    This function scans the stock table,
    deletes any items it finds,
    downloads a CSV file from S3,
    converts the CSV to a DataFrame,
    uploads the DataFrame to the DynamoDB stock table,
    and updates the limit table.

    Args:
        stock_table (Table): The stock table to initialize.
        limit_table (Table): The limit table to update.
        bucket (Bucket): The S3 bucket where the CSV file is stored.
        timezone (tzinfo): The timezone information.
        tmp_dir (str): The temporary directory where the CSV file will be downloaded.
        target_stock (str): The target stock id.
        stock_name (str): The name of the stock.
        period (str): The period of the stock data.
        interval (str): The interval of the stock data.

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
            stock_table.delete_item(Key=key)

    # download original csv from s3
    file_name = f"spp_{stock_config.stock_name}_{stock_config.period}_{stock_config.interval}.csv"
    bucket.download_file(f"csv/{file_name}", f"{lambda_config.tmp_dir}/{file_name}")

    # csv to DataFrame
    original_df: pd.DataFrame = pd.read_csv(
        f"{lambda_config.tmp_dir}/{file_name}", encoding="utf-8", index_col=None
    )

    # convert float to Decimal
    original_df = original_df.apply(
        lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
    )

    # upload to dynamodb stock table
    to_upload_list: list[dict[str, str]] = [
        {str(k): str(v) for k, v in d.items()} for d in original_df.to_dict("records")
    ]
    with stock_table.batch_writer() as batch:
        for item in tqdm(to_upload_list):
            batch.put_item(Item=item)

    # update limit table
    limit_table.put_item(
        Item={
            "stock_id": stock_config.target_stock,
            "operation": "stock",
            "create_at": datetime.now(lambda_config.timezone).isoformat(),
            "max": pd.to_datetime(
                original_df.tail(1)["date"].values[0]
                + " "
                + original_df.tail(1)["time"].values[0]
            ).isoformat(),
        }
    )


def delete_prediction_table(
    prediction_table: Table, limit_table: Table, target_stock: str, timezone: tzinfo
) -> None:
    """
    Delete the prediction table.

    This function scans the prediction table and deletes any items it finds.
    After deleting the items, it updates the limit table with the current operation,
    the creation time, and the maximum number of items.

    Args:
        prediction_table (Table): The prediction table to delete.
        limit_table (Table): The limit table to update.
        target_stock (str): The target stock id.
        timezone (tzinfo): The timezone information.

    Returns:
        None
    """
    # scan
    prediction_df_to_delete = pd.DataFrame(prediction_table.scan()["Items"])

    # delete all items in prediction table
    if not prediction_df_to_delete.empty:
        only_key_to_delete_df = prediction_df_to_delete[["date", "time"]]
        to_delete_list: list[dict[str, str]] = [
            {str(k): str(v) for k, v in d.items()}
            for d in only_key_to_delete_df.to_dict("records")
        ]
        for key in tqdm(to_delete_list):
            prediction_table.delete_item(Key=key)

    # update limit table
    limit_table.put_item(
        Item={
            "stock_id": target_stock,
            "operation": "prediction",
            "create_at": datetime.now(timezone).isoformat(),
            "max": "0",
        }
    )


if __name__ == "__main__":
    load_dotenv(override=True, verbose=True)

    lambda_configuration = LambdaConfiguration(
        pytz.timezone(os.environ["TIMEZONE"]), os.environ["TMP_DIR"]
    )

    stock_configuration = StockConfiguration(
        os.environ["TARGET_STOCK"],
        os.environ["STOCK_NAME"],
        os.environ["PERIOD"],
        os.environ["INTERVAL"],
    )

    aws_auth = AWSAuth(
        os.environ["REGION_NAME"],
        os.environ["ACCESS_KEY_ID"],
        os.environ["SECRET_ACCESS_KEY"],
    )

    # aws instance
    dynamodb: DynamoDBServiceResource = boto3.resource(
        "dynamodb",
        region_name=aws_auth.region_name,
        aws_access_key_id=aws_auth.access_key_id,
        aws_secret_access_key=aws_auth.secret_access_key,
    )

    dynamodb_tables = DynamoDBTables(
        dynamodb.Table(os.environ["AWS_DYNAMODB_STOCK_TABLE_NAME"]),
        dynamodb.Table(os.environ["AWS_DYNAMODB_PREDICTION_TABLE_NAME"]),
        dynamodb.Table(os.environ["AWS_DYNAMODB_LIMIT_TABLE_NAME"]),
    )

    s3: S3ServiceResource = boto3.resource(
        "s3",
        region_name=aws_auth.region_name,
        aws_access_key_id=aws_auth.access_key_id,
        aws_secret_access_key=aws_auth.secret_access_key,
    )

    s3_bucket: Bucket = s3.Bucket(os.environ["AWS_S3_BUCKET_NAME"])

    if "y" == input("Do you want to init limit table? [y/n]"):
        init_limit_table(dynamodb_tables.limit_table)

    if "y" == input("Do you want to init stock table? [y/n]"):
        init_stock_table(
            dynamodb_tables.stock_table,
            dynamodb_tables.limit_table,
            s3_bucket,
            lambda_configuration,
            stock_configuration,
        )

    if "y" == input("Do you want to delete prediction table? [y/n]"):
        delete_prediction_table(
            dynamodb_tables.prediction_table,
            dynamodb_tables.limit_table,
            stock_configuration.target_stock,
            lambda_configuration.timezone,
        )
