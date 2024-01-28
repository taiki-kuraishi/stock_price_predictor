"""
This module contains the main handler function for a AWS Lambda function.

It includes functions to update the stock table, update predictions,
update and predict tables, and update the model.
"""
import os
from datetime import datetime, timedelta, tzinfo
from decimal import Decimal

import boto3
import pandas as pd
import pytz
import yfinance as yf
from aws_lambda_context import LambdaContext
from modules.configurations import (
    AWSAuth,
    DynamoDBTables,
    LambdaConfiguration,
    ModelsConfiguration,
    StockConfiguration,
)
from mypy_boto3_dynamodb import DynamoDBServiceResource
from tqdm import tqdm


def fetch_stock_data(
    timezone_info: tzinfo,
    stock_config: StockConfiguration,
    df_col_order: list[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Fetches and processes stock price data for a given date range and stock symbol.

    This function uses yfinance to download stock price data, and then processes the data.
    The processing includes:
    - Renaming and formatting columns
    - Converting and removing date timezones
    - Excluding data already stored
    - Splitting date and time
    - Converting float values to Decimal type

    Parameters
    ----------
    timezone_info : pytz.timezone
        The timezone information to convert to
    stock_config : StockConfiguration
        The stock price data configuration (target stock symbol, interval, etc.)
    df_col_order : list[str]
        The order of columns for the final dataframe
    start_date : datetime
        The start date for fetching the stock price data
    end_date : datetime
        The end date for fetching the stock price data

    Returns
    -------
    pd.DataFrame
        The processed stock price data
    """
    yfinance_response_df = pd.DataFrame(
        yf.download(
            tickers=stock_config.target_stock,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=stock_config.interval,
        )
    )
    if yfinance_response_df.empty:
        return yfinance_response_df

    yfinance_response_df.reset_index(inplace=True, drop=False)
    yfinance_response_df.columns = yfinance_response_df.columns.str.lower()

    # rename column
    yfinance_response_df.rename(columns={"date": "datetime"}, inplace=True)
    yfinance_response_df.rename(columns={"adj close": "adj_close"}, inplace=True)

    yfinance_response_df["datetime"] = pd.to_datetime(yfinance_response_df["datetime"])

    # convert timezone
    yfinance_response_df["datetime"] = yfinance_response_df["datetime"].dt.tz_convert(
        timezone_info
    )

    # delete timezone
    yfinance_response_df["datetime"] = yfinance_response_df["datetime"].dt.tz_localize(
        None
    )

    # すでに格納されているデータは除外
    yfinance_response_df = yfinance_response_df[
        yfinance_response_df["datetime"] > start_date
    ]

    # datetime to split date and time
    yfinance_response_df["date"] = yfinance_response_df["datetime"].apply(
        lambda x: x.date().isoformat()
    )
    yfinance_response_df["time"] = yfinance_response_df["datetime"].apply(
        lambda x: x.time().isoformat()
    )

    yfinance_response_df = yfinance_response_df.drop(["datetime"], axis=1)
    yfinance_response_df = yfinance_response_df.reindex(columns=df_col_order)

    # float型の値をDecimal型に変換
    yfinance_response_df = yfinance_response_df.apply(
        lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
    )

    return yfinance_response_df


def update_stock_table(
    timezone_info: tzinfo,
    stock_config: StockConfiguration,
    df_col_order: list[str],
    tables: DynamoDBTables,
) -> int:
    """
    Updates the stock table with new data fetched for a given stock symbol.

    This function retrieves the last update date from the limit table,
    fetches new stock data from yfinance,
    and updates the stock and limit tables with the new data.

    Parameters
    ----------
    timezone_info : pytz.timezone
        The timezone information to convert to
    stock_config : StockConfiguration
        The stock price data configuration (target stock symbol, interval, etc.)
    df_col_order : list[str]
        The order of columns for the final dataframe
    tables : DynamoDBTables
        The DynamoDB tables to update

    Returns
    -------
    int
        The number of new records added to the stock table
    """
    # get last update date
    last_update_date = pd.to_datetime(
        str(
            tables.limit_table.get_item(
                Key={"stock_id": stock_config.target_stock, "operation": "stock"}
            )["Item"]["max"]
        )
    )
    end_date = datetime.now(timezone_info) + timedelta(days=1)
    print("\tstart date : " + last_update_date.strftime("%Y-%m-%d"))
    print("\tend date : " + end_date.strftime("%Y-%m-%d"))

    yfinance_response_df = fetch_stock_data(
        timezone_info,
        stock_config,
        df_col_order,
        last_update_date,
        end_date,
    )

    if yfinance_response_df.empty:
        return 0

    # update stock table
    upload_items: list[dict[str, str]] = [
        {str(k): str(v) for k, v in d.items()}
        for d in yfinance_response_df.to_dict("records")
    ]
    with tables.stock_table.batch_writer() as batch:
        for item in tqdm(upload_items):
            batch.put_item(Item=item)

    # update limit table
    tables.limit_table.put_item(
        Item={
            "stock_id": stock_config.target_stock,
            "operation": "stock",
            "create_at": datetime.now(timezone_info).isoformat(),
            "max": pd.to_datetime(
                yfinance_response_df.tail(1)["date"].values[0]
                + " "
                + yfinance_response_df.tail(1)["time"].values[0]
            ).isoformat(),
        }
    )

    return len(yfinance_response_df)


def handler(event: dict[str, str], _context: LambdaContext) -> dict[str, str]:
    """
    Main handler function for AWS Lambda.

    This function performs one of the following operations based on the 'handler' key in the event:
    - Update the stock price table
    - Update the predictions
    - Update both the stock price table and the prediction table
    - Update the models

    If the 'handler' key does not correspond to any of the above,
    it returns a "wrong handler" message.

    Parameters
    ----------
    event : dict[str, str]
        The event data received from AWS Lambda.
        It should contain a 'handler' key with the operation name.
    _context : LambdaContext
        The context information received from AWS Lambda.
        It is not used in this function.

    Returns
    -------
    dict
        A dictionary containing a message indicating the result of the operation,
        and the number of updated records (if applicable).
    """
    # check handler
    event_handler = event.get("handler", None)
    if event_handler is None or isinstance(event_handler, str) is False:
        raise ValueError("handler is None")

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

    models_configuration = ModelsConfiguration(
        int(os.environ["MODEL_NUM"]),
        os.environ["DATAFRAME_COLUMNS_ORDER"].split(","),
        os.environ["FEATURES_COLUMNS"].split(","),
        os.environ["TARGET_COLUM"],
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

    if event_handler == "update_stock_table":
        return {
            "message": "update_stock_table",
            "update_num": str(
                update_stock_table(
                    lambda_configuration.timezone,
                    stock_configuration,
                    models_configuration.dataframe_columns_order,
                    dynamodb_tables,
                )
            ),
        }

    return {"message": "wrong handler"}
