import os
import pytz
from datetime import datetime, timedelta
from dateutil.parser import parse
from aws_lambda_context import LambdaContext
from modules.dynamodb_fetcher import (
    get_data_from_dynamodb,
    get_data_from_dynamodb_stream,
    get_data_from_dynamodb_stream_while,
    get_data_from_dynamodb_query,
)


def lambda_handler(event: dict, context: LambdaContext):
    # load env
    stock_name: str = os.environ["STOCK_NAME"]
    region_name: str = os.environ["REGION_NAME"]
    access_key_id: str = os.environ["ACCESS_KEY_ID"]
    secret_access_key: str = os.environ["SECRET_ACCESS_KEY"]
    time_list: list[int] = [int(i) for i in os.environ["TIME_LIST"].split(",")]
    dynamodb_pred_table_name = "spp_" + stock_name + "_pred"

    # set timezone
    JST = pytz.timezone("Asia/Tokyo")

    if "handler" not in event:
        return {
            "message": "bad request",
        }

    # hello world
    if event["handler"] == "hello":
        return {
            "message": "hello world",
        }

    # latest
    elif event["handler"] == "latest":
        # get pred data from dynamodb
        pred_df = get_data_from_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_pred_table_name,
        )

        # sort datetime
        pred_df = pred_df.sort_values("datetime", ascending=False)

        # sort columns
        pred_df = pred_df.reindex(sorted(pred_df.columns), axis=1)

        # get max datetime row
        latest_dict = pred_df.head(1).to_dict("records")[0]

        # to type datetime
        dt = parse(latest_dict["datetime"])

        # create response body
        response_body = {
            "prediction_timestamp": latest_dict["datetime"],
            "prediction": {},
        }

        # dict drop id, datetime
        latest_dict.pop("id")
        latest_dict.pop("datetime")

        # create prediction dict
        for key, value in latest_dict.items():
            index = key.split("_")[0]
            pred_time = dt + timedelta(hours=int(index))
            pred_time = pred_time.isoformat()
            response_body["prediction"][pred_time] = str(value)

        print(response_body)

        return response_body

    elif event["handler"] == "latest_stream":
        # get pred data from dynamodb
        dynamodb_response = get_data_from_dynamodb_stream(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_pred_table_name,
        )

        if dynamodb_response is None:
            raise Exception("dynamodb_stream_response is None")

        newImage = dynamodb_response["NewImage"]
        CreationDateTime = dynamodb_response["ApproximateCreationDateTime"].astimezone(
            JST
        )

        # create response body
        response_body = {
            "prediction_timestamp": CreationDateTime.isoformat(),
            "prediction": {},
        }

        # dict drop id, datetime
        newImage.pop("id")
        newImage.pop("datetime")

        # CreationDateTimeの時間のみ取得
        CreationHour = CreationDateTime.hour

        # CreationDateTimeの時間を0にする
        predictionBaseTime = CreationDateTime.replace(minute=0, second=0)

        # predictionBaseTimeからhourを取得
        predictionBaseHour = predictionBaseTime.hour

        # CreationDateTimeの時間がtime_listにない場合はエラー
        if predictionBaseHour not in time_list:
            raise Exception("predictionBaseHour not in time_list")

        time_list_index = time_list.index(predictionBaseHour)

        # create prediction dict
        for key, value in newImage.items():
            index = key.split("_")[0]
            pred_time = predictionBaseTime
            if time_list_index + int(index) >= len(time_list):
                pred_time = pred_time.replace(day=pred_time.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if pred_time.weekday() == 5:
                    pred_time = pred_time.replace(day=pred_time.day + 2)
                elif pred_time.weekday() == 6:
                    pred_time = pred_time.replace(day=pred_time.day + 1)

                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index) - len(time_list)])
                )
            else:
                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index)])
                )
            pred_time = pred_time.isoformat()
            response_body["prediction"][str(index) + "_hour_prediction"] = {
                "value": str(value["N"]),
                "datetime": pred_time,
            }

        print("response_body : ", response_body)

        return response_body

    elif event["handler"] == "latest_stream_while":
        # get pred data from dynamodb
        dynamodb_response = get_data_from_dynamodb_stream_while(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_pred_table_name,
        )

        print("dynamodb_response : ", dynamodb_response)

        if dynamodb_response is None:
            raise Exception("dynamodb_stream_response is None")

        newImage = dynamodb_response["NewImage"]
        CreationDateTime = dynamodb_response["ApproximateCreationDateTime"].astimezone(
            JST
        )

        # create response body
        response_body = {
            "prediction_timestamp": CreationDateTime.isoformat(),
            "prediction": {},
        }

        # dict drop id, datetime
        newImage.pop("id")
        newImage.pop("datetime")

        # CreationDateTimeの時間のみ取得
        CreationHour = CreationDateTime.hour

        # CreationDateTimeの時間を0にする
        predictionBaseTime = CreationDateTime.replace(minute=0, second=0)

        # predictionBaseTimeからhourを取得
        predictionBaseHour = predictionBaseTime.hour

        # CreationDateTimeの時間がtime_listにない場合はエラー
        if predictionBaseHour not in time_list:
            print("time_list : ", time_list)
            print("predictionBaseHour : ", predictionBaseHour)
            raise Exception("predictionBaseHour not in time_list")

        time_list_index = time_list.index(predictionBaseHour)

        # create prediction dict
        for key, value in newImage.items():
            index = key.split("_")[0]
            pred_time = predictionBaseTime
            if time_list_index + int(index) >= len(time_list):
                pred_time = pred_time.replace(day=pred_time.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if pred_time.weekday() == 5:
                    pred_time = pred_time.replace(day=pred_time.day + 2)
                elif pred_time.weekday() == 6:
                    pred_time = pred_time.replace(day=pred_time.day + 1)

                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index) - len(time_list)])
                )
            else:
                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index)])
                )
            pred_time = pred_time.isoformat()
            response_body["prediction"][str(index) + "_hour_prediction"] = {
                "value": str(value["N"]),
                "datetime": pred_time,
            }

        print("response_body : ", response_body)

        return response_body

    elif event["handler"] == "latest_query":
        """
        dynamodbからqueryを使用してデータを取得する
        9:00以前の場合は前日の14:00から取得する
        時間が9~14でなおかつ分が20分以下の場合は1時間前のデータを取得する
        14:00以降の場合は当日14:00のデータを取得する
        """
        dynamodb_pred_table_name = "spp_prediction"

        datetime_now = datetime.now(JST).replace(second=0, microsecond=0)

        if datetime_now < datetime_now.replace(
            hour=9,
            minute=20,
        ):
            datetime_now = datetime_now.replace(
                day=datetime_now.day - 1,
                hour=14,
                minute=0,
            )
        elif datetime_now.hour in time_list and datetime_now.minute <= 20:
            datetime_now = datetime_now.replace(
                hour=datetime_now.hour - 1,
                minute=0,
            )
        elif datetime_now > datetime_now.replace(hour=14, minute=20):
            datetime_now = datetime_now.replace(
                hour=14,
                minute=0,
            )

        iso_formatted_datetime_now = datetime_now.isoformat().split("+")[0]
        prediction_df = get_data_from_dynamodb_query(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_pred_table_name,
            iso_formatted_datetime_now,
        )

        if prediction_df.empty:
            prediction_df = get_data_from_dynamodb(
                region_name,
                access_key_id,
                secret_access_key,
                dynamodb_pred_table_name,
            )
            prediction_df = prediction_df.loc[prediction_df["datetime"].idxmax()]
            prediction_dict = prediction_df.to_dict()
        else:
            prediction_dict = prediction_df.to_dict("records")[0]

        print(prediction_dict)

        creation_datetime = datetime.fromisoformat(
            prediction_dict["creation_datetime"]
        )
        print(creation_datetime)

        response_body = {
            "prediction_timestamp": creation_datetime.isoformat(),
            "prediction": {},
        }

        prediction_dict.pop("datetime")
        prediction_dict.pop("creation_datetime")

        predictionBaseTime = creation_datetime.replace(minute=0, second=0)

        # predictionBaseTimeからhourを取得
        predictionBaseHour = predictionBaseTime.hour

        # creation_datetimeの時間がtime_listにない場合はエラー
        if predictionBaseHour not in time_list:
            print("time_list : ", time_list)
            print("predictionBaseHour : ", predictionBaseHour)
            raise Exception("predictionBaseHour not in time_list")

        time_list_index = time_list.index(predictionBaseHour)

        # create prediction dict
        for key, value in prediction_dict.items():
            index = key.split("_")[0]
            pred_time = predictionBaseTime
            if time_list_index + int(index) >= len(time_list):
                pred_time = pred_time.replace(day=pred_time.day + 1)

                # weakDayが5,6(土、日)の場合は月曜日まで日付を進める
                if pred_time.weekday() == 5:
                    pred_time = pred_time.replace(day=pred_time.day + 2)
                elif pred_time.weekday() == 6:
                    pred_time = pred_time.replace(day=pred_time.day + 1)

                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index) - len(time_list)])
                )
            else:
                pred_time = pred_time.replace(
                    hour=int(time_list[time_list_index + int(index)])
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
