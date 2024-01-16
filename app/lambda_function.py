import os
import boto3
import pytz
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Key
from aws_lambda_context import LambdaContext


def lambda_handler(event: dict, context: LambdaContext):
    # load env
    timezone: str = os.environ["TIMEZONE"]
    stock_name: str = os.environ["STOCK_NAME"]
    region_name: str = os.environ["REGION_NAME"]
    access_key_id: str = os.environ["ACCESS_KEY_ID"]
    secret_access_key: str = os.environ["SECRET_ACCESS_KEY"]
    time_list: list[int] = [int(i) for i in os.environ["TIME_LIST"].split(",")]
    dict_order: list[str] = os.environ["DICT_ORDER"].split(",")
    dynamodb_pred_table_name = "stock_price_predictor_" + stock_name + "_prediction"

    # set timezone
    JST = pytz.timezone(timezone)

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

    # hello world
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
        prediction_dict = {key: prediction_dict[key] for key in dict_order}

        response_body = {
            "prediction_timestamp": prediction_dict["create_at"],
            "prediction": {},
        }

        # predictionBaseTimeからhourを取得
        predictionBaseHour = prediction_dict["datetime"].hour

        # creation_datetimeの時間がtime_listにない場合はエラー
        if predictionBaseHour not in time_list:
            print("time_list : ", time_list)
            print("predictionBaseHour : ", predictionBaseHour)
            raise Exception("predictionBaseHour not in time_list")

        time_list_index = time_list.index(predictionBaseHour)

        # create prediction dict
        for key, value in prediction_dict.items():
            if (
                key == "date"
                or key == "time"
                or key == "datetime"
                or key == "create_at"
            ):
                continue

            index = int(key.split("_")[0])
            pred_time = prediction_dict["datetime"]

            if time_list_index + index >= len(time_list):
                pred_time = pred_time.replace(day=pred_time.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if pred_time.weekday() == 5:
                    pred_time = pred_time.replace(day=pred_time.day + 2)
                elif pred_time.weekday() == 6:
                    pred_time = pred_time.replace(day=pred_time.day + 1)

                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + index - len(time_list)])
                )
            else:
                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + index])
                )
            pred_time = pred_time.isoformat()
            response_body["prediction"][str(index) + "_hour_prediction"] = {
                "value": str(value),
                "datetime": pred_time,
            }

        print("response_body : ", response_body)
        return response_body

    # error
    else:
        return {
            "message": "bad request",
        }
