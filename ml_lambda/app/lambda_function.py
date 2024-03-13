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
from aws_lambda_context import (
    LambdaClientContext,
    LambdaClientContextMobileClient,
    LambdaCognitoIdentity,
    LambdaContext,
)
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv
from joblib import dump, load
from modules.configurations import (
    AWSAuth,
    DynamoDBTables,
    LambdaConfiguration,
    ModelsConfiguration,
    StockConfiguration,
)
from modules.data_preprocessing import (
    post_process_stock_data_from_dynamodb,
    shift_dataframe,
)
from mypy_boto3_dynamodb import DynamoDBServiceResource
from mypy_boto3_s3 import S3ServiceResource
from mypy_boto3_s3.service_resource import Bucket
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


def generate_predict(
    model_num: int,
    stock_name: str,
    tmp_dir: str,
    bucket: Bucket,
    x: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates stock price predictions using the specified model
    and returns a DataFrame containing the predictions.

    Parameters:
    model_num (int): The number of the model to use,
    which is included in the model's file name.
    stock_name (str): The name of the stock for which to generate predictions,
    which is included in the model's file name.
    tmp_dir (str): The path to the directory
    where the model file will be temporarily saved.
    bucket (Bucket): The S3 bucket from which to download the model file.
    x (pd.DataFrame): The input data for generating predictions.

    Returns:
    pd.DataFrame: A DataFrame containing the predictions.
    Each prediction has the model's number as its column name.
    """
    model_file_name = (
        f"spp_{stock_name}_{str(model_num)}h_PassiveAggressiveRegressor.pkl"
    )
    model_local_path = os.path.join(tmp_dir, model_file_name)
    bucket.download_file(f"models/{model_file_name}", model_local_path)
    model = load(model_local_path)
    prediction = pd.DataFrame()
    prediction[str(model_num)] = model.predict(x)
    return prediction


def update_predict(
    lambda_config: LambdaConfiguration,
    stock_config: StockConfiguration,
    models_config: ModelsConfiguration,
    tables: DynamoDBTables,
    bucket: Bucket,
) -> int:
    """
    Updates the stock price predictions.

    Parameters
    ----------
    lambda_config : LambdaConfiguration
        The configuration for the Lambda function.
    stock_config : StockConfiguration
        The configuration for the stock data.
    models_config : ModelsConfiguration
        The configuration for the models.
    tables : DynamoDBTables
        The DynamoDB tables.
    bucket : Bucket
        The S3 bucket.

    Returns
    -------
    int
        The number of updated predictions.
    """
    # get last update date
    last_prediction_update_datetime = pd.to_datetime(
        str(
            tables.limit_table.get_item(
                Key={"stock_id": stock_config.target_stock, "operation": "prediction"}
            )["Item"]["max"]
        )
    )

    if last_prediction_update_datetime == "0":
        unpredicted_df = pd.DataFrame(tables.stock_table.scan()["Items"])

    if last_prediction_update_datetime != "0":
        last_stock_update_datetime = str(
            tables.limit_table.get_item(
                Key={"stock_id": stock_config.target_stock, "operation": "stock"}
            )["Item"]["max"]
        )

        unpredicted_df = pd.DataFrame()
        date = last_prediction_update_datetime.strftime("%Y-%m-%d")
        while date <= last_stock_update_datetime:
            day_of_prediction_df = pd.DataFrame(
                tables.stock_table.query(KeyConditionExpression=Key("date").eq(date))[
                    "Items"
                ]
            )
            # すでに予測済みのデータは除外
            if date == last_prediction_update_datetime.strftime("%Y-%m-%d"):
                day_of_prediction_df = day_of_prediction_df[
                    day_of_prediction_df["time"]
                    > last_prediction_update_datetime.strftime("%H:%M:%S")
                ]
            unpredicted_df = pd.concat([unpredicted_df, day_of_prediction_df])
            date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

    if unpredicted_df.empty:
        return 0

    # カラムのソートと日付でソート
    unpredicted_df = post_process_stock_data_from_dynamodb(
        unpredicted_df, models_config.dataframe_columns_order
    )

    # 新たに予測したデータを格納するdfの作成
    update_prediction_df = pd.DataFrame()
    update_prediction_df["date"] = unpredicted_df[["date"]]
    update_prediction_df["time"] = unpredicted_df[["time"]]

    for i in tqdm(range(1, models_config.models_number + 1)):
        update_prediction_df = pd.concat(
            [
                update_prediction_df,
                generate_predict(
                    i,
                    stock_config.stock_name,
                    lambda_config.tmp_dir,
                    bucket,
                    unpredicted_df[models_config.features_columns],
                ),
            ],
            axis=1,
        )

    # convert float to decimal
    update_prediction_df = update_prediction_df.apply(
        lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
    )

    # create_at column
    update_prediction_df["create_at"] = datetime.now(lambda_config.timezone).isoformat()

    print(update_prediction_df)

    # update prediction table
    upload_items: list[dict[str, str | Decimal]] = [
        {str(k): Decimal(v) if str(k).isdigit() else str(v) for k, v in d.items()}
        for d in update_prediction_df.to_dict("records")
    ]
    with tables.prediction_table.batch_writer() as batch:
        for item in tqdm(upload_items):
            batch.put_item(Item=item)

    # update limit table
    tables.limit_table.put_item(
        Item={
            "stock_id": stock_config.target_stock,
            "operation": "prediction",
            "create_at": datetime.now(lambda_config.timezone).isoformat(),
            "max": pd.to_datetime(
                update_prediction_df.tail(1)["date"].values[0]
                + " "
                + update_prediction_df.tail(1)["time"].values[0]
            ).isoformat(),
        }
    )

    return len(update_prediction_df)


def update_model(
    lambda_config: LambdaConfiguration,
    stock_config: StockConfiguration,
    models_config: ModelsConfiguration,
    tables: DynamoDBTables,
    bucket: Bucket,
) -> int:
    """
    Updates the model and the limit table in DynamoDB.

    This function partially fits the model for the specified stock
    using data from the last trained date to the last stock update date.
    The fitted model is then uploaded to an S3 bucket. Additionally,
    a new entry is added to the limit table to update the last trained date.

    Parameters:
    lambda_config (LambdaConfiguration): The configuration for the lambda function.
    stock_config (StockConfiguration): The configuration for the stock.
    models_config (ModelsConfiguration): The configuration for the models.
    tables (DynamoDBTables): The DynamoDB tables.
    bucket (Bucket): The S3 bucket.

    Returns:
    int: The length of the untrained dataframe.
    """
    last_train_date = (
        pd.to_datetime(
            str(
                tables.limit_table.get_item(
                    Key={"stock_id": stock_config.target_stock, "operation": "train"}
                )["Item"]["max"]
            )
        )
        + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    last_stock_update_date = pd.to_datetime(
        str(
            tables.limit_table.get_item(
                Key={"stock_id": stock_config.target_stock, "operation": "stock"}
            )["Item"]["max"]
        )
    ).strftime("%Y-%m-%d")

    print("\tlast train date : " + last_train_date)
    print("\tlast stock update date : " + last_stock_update_date)

    untrained_df = pd.DataFrame()
    date = last_train_date
    while date <= last_stock_update_date:
        day_of_train_df = pd.DataFrame(
            tables.stock_table.query(KeyConditionExpression=Key("date").eq(date))[
                "Items"
            ]
        )
        untrained_df = pd.concat([untrained_df, day_of_train_df])
        date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

    if untrained_df.empty:
        return 0

    # カラムのソートと日付でソート
    untrained_df = post_process_stock_data_from_dynamodb(
        untrained_df, models_config.dataframe_columns_order
    )

    # model update
    for i in tqdm(range(1, models_config.models_number + 1)):
        model_file_name = (
            f"spp_{stock_config.stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl"
        )
        model_local_path = f"{lambda_config.tmp_dir}/{model_file_name}"
        bucket.download_file(f"models/{model_file_name}", model_local_path)
        train_df = shift_dataframe(
            df=untrained_df, shift_rows=i, target_col=models_config.target_column
        )
        model = load(model_local_path)
        model.partial_fit(
            train_df[models_config.features_columns],
            train_df[models_config.target_column],
        )
        dump(model, model_local_path)
        bucket.upload_file(model_local_path, f"models/{model_file_name}")

    # update limit table
    tables.limit_table.put_item(
        Item={
            "stock_id": stock_config.target_stock,
            "operation": "train",
            "create_at": datetime.now(lambda_config.timezone).isoformat(),
            "max": last_stock_update_date,
        }
    )

    return len(untrained_df)


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

    s3: S3ServiceResource = boto3.resource(
        "s3",
        region_name=aws_auth.region_name,
        aws_access_key_id=aws_auth.access_key_id,
        aws_secret_access_key=aws_auth.secret_access_key,
    )

    s3_bucket: Bucket = s3.Bucket(os.environ["AWS_S3_BUCKET_NAME"])

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

    if event_handler == "update_predict":
        return {
            "message": "update_predict",
            "update_num": str(
                update_predict(
                    lambda_configuration,
                    stock_configuration,
                    models_configuration,
                    dynamodb_tables,
                    s3_bucket,
                )
            ),
        }

    if event_handler == "update_and_predict_tables":
        return {
            "message": "update_and_predict_tables",
            "update_stock_num": str(
                update_stock_table(
                    lambda_configuration.timezone,
                    stock_configuration,
                    models_configuration.dataframe_columns_order,
                    dynamodb_tables,
                )
            ),
            "update_predict_num": str(
                update_predict(
                    lambda_configuration,
                    stock_configuration,
                    models_configuration,
                    dynamodb_tables,
                    s3_bucket,
                )
            ),
        }

    if event_handler == "update_model":
        return {
            "message": "update_model",
            "update_num": str(
                update_model(
                    lambda_configuration,
                    stock_configuration,
                    models_configuration,
                    dynamodb_tables,
                    s3_bucket,
                )
            ),
        }

    return {"message": "wrong handler"}


if __name__ == "__main__":
    load_dotenv(dotenv_path="ml_lambda/.env.ml", override=True, verbose=True)

    lambda_cognito_identity = LambdaCognitoIdentity()
    lambda_cognito_identity.cognito_identity_id = "cognito_identity_id"
    lambda_cognito_identity.cognito_identity_pool_id = "cognito_identity_pool_id"

    lambda_client_context_mobile_client = LambdaClientContextMobileClient()
    lambda_client_context_mobile_client.installation_id = "installation_id"
    lambda_client_context_mobile_client.app_title = "app_title"
    lambda_client_context_mobile_client.app_version_name = "app_version_name"
    lambda_client_context_mobile_client.app_version_code = "app_version_code"
    lambda_client_context_mobile_client.app_package_name = "app_package_name"

    lambda_client_context = LambdaClientContext()
    lambda_client_context.client = lambda_client_context_mobile_client
    lambda_client_context.custom = {"custom": True}
    lambda_client_context.env = {"env": "test"}

    lambda_context = LambdaContext()
    lambda_context.function_name = "function_name"
    lambda_context.function_version = "function_version"
    lambda_context.invoked_function_arn = "invoked_function_arn"
    lambda_context.memory_limit_in_mb = 300
    lambda_context.aws_request_id = "aws_request_id"
    lambda_context.log_group_name = "log_group_name"
    lambda_context.log_stream_name = "log_stream_name"
    lambda_context.identity = lambda_cognito_identity
    lambda_context.client_context = lambda_client_context

    handler({"handler": ""}, lambda_context)
    print(handler({"handler": "update_predict"}, lambda_context))
    print(handler({"handler": "update_stock_table"}, lambda_context))
    print(handler({"handler": "update_and_predict_tables"}, lambda_context))
    print(handler({"handler": "update_model"}, lambda_context))
