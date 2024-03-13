"""
This module is used to predict stock prices using AWS Lambda and DynamoDB.
"""

import os
from datetime import datetime

import boto3
import pandas as pd
import pytz
from aws_lambda_context import (
    LambdaClientContext,
    LambdaClientContextMobileClient,
    LambdaCognitoIdentity,
    LambdaContext,
)
from boto3.dynamodb.conditions import Key
from dataclass.dataclass import StockConfiguration
from dotenv import load_dotenv
from mypy_boto3_dynamodb.service_resource import Table


def fetch_latest_prediction(
    stock_configuration: StockConfiguration,
    table: Table,
) -> dict[str, str | dict[str, dict[str, str]]]:
    """
    Fetches the latest stock price prediction from a DynamoDB table.

    This function queries a DynamoDB table for the latest prediction for the current date.
    If no prediction is found for the current date,
    it scans the entire table and retrieves the latest prediction.

    The function then formats the prediction into a response body,
    which includes the prediction timestamp and the prediction itself.

    Parameters
    ----------
    lambda_configuration : LambdaConfiguration
        The configuration for the AWS Lambda function.
    stock_configuration : StockConfiguration
        The configuration for the stock price prediction.
    table : Table
        The DynamoDB table from which to fetch the prediction.

    Returns
    -------
    dict[str, str]
        A dictionary containing the prediction timestamp and the prediction.
    """
    todays_date = datetime.now(stock_configuration.timezone).strftime("%Y-%m-%d")

    response = table.query(KeyConditionExpression=Key("date").eq(todays_date))["Items"]

    if len(response) != 0:
        # datetimeでソート
        response_df = pd.DataFrame(response)
        response_df["datetime"] = pd.to_datetime(
            response_df["date"] + " " + response_df["time"]
        )
        response_df = response_df.sort_values("datetime", ascending=True)
        prediction_dict = response_df.tail(1).to_dict("records")[0]
    else:
        response_df = pd.DataFrame(table.scan()["Items"])
        response_df["datetime"] = pd.to_datetime(
            response_df["date"] + " " + response_df["time"]
        )
        response_df = response_df.sort_values("datetime", ascending=True)
        prediction_dict = response_df.tail(1).to_dict("records")[0]

    # sort dict key
    prediction_dict = {
        key: prediction_dict[key] for key in stock_configuration.dict_order
    }

    response_body: dict[str, str | dict[str, dict[str, str]]] = {
        "prediction_timestamp": prediction_dict["create_at"],
        "prediction": {},
    }

    # predictionBaseTimeからhourを取得
    prediction_base_hour = prediction_dict["datetime"].hour

    # creation_datetimeの時間がtime_listにない場合は、最も近い時間を取得
    if prediction_base_hour not in stock_configuration.time_list:
        # time_listの中で最も近い時間を取得
        prediction_base_hour = min(
            stock_configuration.time_list, key=lambda x: abs(x - prediction_base_hour)
        )

    time_list_index = stock_configuration.time_list.index(prediction_base_hour)

    # pred_time
    pred_time = prediction_dict["datetime"]
    print("pred_time : ", pred_time)

    # drop date,time,datetime, create_at
    prediction_dict.pop("date")
    prediction_dict.pop("time")
    prediction_dict.pop("datetime")
    prediction_dict.pop("create_at")

    # create prediction dict
    for key, value in prediction_dict.items():
        datetime_value = pred_time

        if time_list_index + int(str(key)) >= len(stock_configuration.time_list):
            datetime_value = datetime_value.replace(day=datetime_value.day + 1)

            # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
            if datetime_value.weekday() == stock_configuration.saturday:
                datetime_value = datetime_value.replace(day=datetime_value.day + 2)
            elif datetime_value.weekday() == stock_configuration.sunday:
                datetime_value = datetime_value.replace(day=datetime_value.day + 1)

            datetime_value = datetime_value.replace(
                hour=int(
                    stock_configuration.time_list[
                        time_list_index
                        + int(str(key))
                        - len(stock_configuration.time_list)
                    ]
                )
            )
        else:
            datetime_value = datetime_value.replace(
                hour=int(stock_configuration.time_list[time_list_index + int(str(key))])
            )

        if isinstance(response_body["prediction"], dict):
            response_body["prediction"][str(key) + "_hour_prediction"] = {
                "value": str(value),
                "datetime": datetime_value.isoformat(),
            }

    print("response_body : ", response_body)
    return response_body


def lambda_handler(
    event: dict[str, str], _context: LambdaContext
) -> dict[str, str | dict[str, dict[str, str]]]:
    """
    Main handler function for AWS Lambda.

    This function performs the following operations based on the 'handler' key in the event:
    - If the 'handler' key is 'latest', it retrieves the latest stock price prediction.

    The function first loads environment variables and sets up AWS resources.
    It then checks the 'handler' key in the event.
    If the 'handler' key is 'latest',
    it queries the DynamoDB table for the latest prediction for the current date.
    If no prediction is found for the current date,
    it scans the entire table and retrieves the latest prediction.

    The function then formats the prediction into a response body,
    which includes the prediction timestamp and the prediction itself.

    If the 'handler' key does not correspond to 'latest',
    it returns a "bad request" message.

    Parameters
    ----------
    event : dict
        The event data received from AWS Lambda.
        It should contain a 'handler' key with the operation name.
    _context : LambdaContext
        The context information received from AWS Lambda.
        It is not used in this function.

    Returns
    -------
    dict[str, str]
        A dictionary containing the prediction timestamp and the prediction,
        or a message indicating a bad request.
    """

    stock_configuration = StockConfiguration(
        pytz.timezone(os.environ["TIMEZONE"]),
        int(os.environ["SATURDAY"]),
        int(os.environ["SUNDAY"]),
        [int(i) for i in os.environ["TIME_LIST"].split(",")],
        os.environ["DICT_ORDER"].split(","),
    )

    # aws instance
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"],
    )
    table = dynamodb.Table(
        "stock_price_predictor_" + os.environ["STOCK_NAME"] + "_prediction"
    )

    if "handler" not in event:
        return {
            "message": "bad request",
        }

    if event["handler"] == "latest":
        return fetch_latest_prediction(
            stock_configuration,
            table,
        )

    return {
        "message": "bad request",
    }


if __name__ == "__main__":
    load_dotenv(dotenv_path="api_lambda/.env.api", verbose=True, override=True)

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

    print(lambda_handler({"handler": "latest"}, lambda_context))
