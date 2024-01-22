import os
import boto3
import pytz
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Key
from aws_lambda_context import LambdaContext


def lambda_handler(event: dict, context: LambdaContext):
    # load env
    SATURDAY: int = os.environ["SATURDAY"]
    SUNDAY: int = os.environ["SUNDAY"]
    TIMEZONE: str = os.environ["TIMEZONE"]
    STOCK_NAME: str = os.environ["STOCK_NAME"]
    REGION_NAME: str = os.environ["REGION_NAME"]
    ACCESS_KEY_ID: str = os.environ["ACCESS_KEY_ID"]
    SECRET_ACCESS_KEY: str = os.environ["SECRET_ACCESS_KEY"]
    TIME_LIST: list[int] = [int(i) for i in os.environ["TIME_LIST"].split(",")]
    DICT_ORDER: list[str] = os.environ["DICT_ORDER"].split(",")
    DYNAMODB_PRED_TABLE_NAME: str = (
        "stock_price_predictor_" + STOCK_NAME + "_prediction"
    )

    # set timezone
    JST = pytz.timezone(TIMEZONE)

    # aws instance
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=REGION_NAME,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )
    table = dynamodb.Table(DYNAMODB_PRED_TABLE_NAME)

    if "handler" not in event:
        return {
            "message": "bad request",
        }

    if event["handler"] == "latest":
        todays_date = datetime.now(JST).strftime("%Y-%m-%d")

        response = table.query(KeyConditionExpression=Key("date").eq(todays_date))[
            "Items"
        ]

        if len(response) != 0:
            # datetimeでソート
            response_df = pd.DataFrame(response)
            response_df["datetime"] = pd.to_datetime(
                response_df["date"] + " " + response_df["time"]
            )
            response_df = response_df.sort_values("datetime", ascending=True)
            prediction_dict = response_df.tail(1).to_dict("records")[0]
        else:
            response = table.scan()
            response_df = pd.DataFrame(response["Items"])
            response_df["datetime"] = pd.to_datetime(
                response_df["date"] + " " + response_df["time"]
            )
            response_df = response_df.sort_values("datetime", ascending=True)
            prediction_dict = response_df.tail(1).to_dict("records")[0]

        # sort dict key
        prediction_dict = {key: prediction_dict[key] for key in DICT_ORDER}

        response_body = {
            "prediction_timestamp": prediction_dict["create_at"],
            "prediction": {},
        }

        # predictionBaseTimeからhourを取得
        predictionBaseHour = prediction_dict["datetime"].hour

        # creation_datetimeの時間がtime_listにない場合は、最も近い時間を取得
        if predictionBaseHour not in TIME_LIST:
            # time_listの中で最も近い時間を取得
            predictionBaseHour = min(
                TIME_LIST, key=lambda x: abs(x - predictionBaseHour)
            )

        time_list_index = TIME_LIST.index(predictionBaseHour)

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

            if time_list_index + int(key) >= len(TIME_LIST):
                datetime_value = datetime_value.replace(day=datetime_value.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if datetime_value.weekday() == SATURDAY:
                    datetime_value = datetime_value.replace(day=datetime_value.day + 2)
                elif datetime_value.weekday() == SUNDAY:
                    datetime_value = datetime_value.replace(day=datetime_value.day + 1)

                datetime_value = datetime_value.replace(
                    hour=int(TIME_LIST[time_list_index + int(key) - len(TIME_LIST)])
                )
            else:
                datetime_value = datetime_value.replace(
                    hour=int(TIME_LIST[time_list_index + int(key)])
                )

            response_body["prediction"][str(int(key)) + "_hour_prediction"] = {
                "value": str(value),
                "datetime": datetime_value.isoformat(),
            }

        print("response_body : ", response_body)
        return response_body

    # error
    else:
        return {
            "message": "bad request",
        }
