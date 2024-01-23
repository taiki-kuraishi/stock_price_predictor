import os
import boto3
import pytz
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Key
from aws_lambda_context import LambdaContext


def lambda_handler(event: dict, context: LambdaContext):
    # load env
    saturday: int = os.environ["SATURDAY"]
    sunday: int = os.environ["SUNDAY"]
    timezone: str = os.environ["TIMEZONE"]
    stock_name: str = os.environ["STOCK_NAME"]
    region_name: str = os.environ["REGION_NAME"]
    access_key_id: str = os.environ["ACCESS_KEY_ID"]
    secret_access_key: str = os.environ["SECRET_ACCESS_KEY"]
    time_list: list[int] = [int(i) for i in os.environ["TIME_LIST"].split(",")]
    dict_order: list[str] = os.environ["DICT_ORDER"].split(",")
    dynamodb_pred_table_name: str = (
        "stock_price_predictor_" + stock_name + "_prediction"
    )

    # set timezone
    jst = pytz.timezone(timezone)

    # aws instance
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )
    table = dynamodb.Table(dynamodb_pred_table_name)

    if "handler" not in event:
        return {
            "message": "bad request",
        }

    if event["handler"] == "latest":
        todays_date = datetime.now(jst).strftime("%Y-%m-%d")

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
        prediction_dict = {key: prediction_dict[key] for key in dict_order}

        response_body = {
            "prediction_timestamp": prediction_dict["create_at"],
            "prediction": {},
        }

        # predictionBaseTimeからhourを取得
        predictionBaseHour = prediction_dict["datetime"].hour

        # creation_datetimeの時間がtime_listにない場合は、最も近い時間を取得
        if predictionBaseHour not in time_list:
            # time_listの中で最も近い時間を取得
            predictionBaseHour = min(
                time_list, key=lambda x: abs(x - predictionBaseHour)
            )

        time_list_index = time_list.index(predictionBaseHour)

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

            if time_list_index + int(key) >= len(time_list):
                datetime_value = datetime_value.replace(day=datetime_value.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if datetime_value.weekday() == saturday:
                    datetime_value = datetime_value.replace(day=datetime_value.day + 2)
                elif datetime_value.weekday() == sunday:
                    datetime_value = datetime_value.replace(day=datetime_value.day + 1)

                datetime_value = datetime_value.replace(
                    hour=int(time_list[time_list_index + int(key) - len(time_list)])
                )
            else:
                datetime_value = datetime_value.replace(
                    hour=int(time_list[time_list_index + int(key)])
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
