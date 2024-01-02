import os
import json
from datetime import timedelta
from dateutil.parser import parse
from aws_lambda_context import LambdaContext
from modules.dynamodb_fetcher import get_data_from_dynamodb


def lambda_handler(event: dict, context: LambdaContext):
    # load env
    stock_name: str = os.environ["STOCK_NAME"]
    region_name: str = os.environ["REGION_NAME"]
    access_key_id: str = os.environ["ACCESS_KEY_ID"]
    secret_access_key: str = os.environ["SECRET_ACCESS_KEY"]
    dynamodb_pred_table_name = "spp_" + stock_name + "_pred"

    if "handler" not in event:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "bad request",
                }
            ),
        }

    # hello world
    if event["handler"] == "hello":
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "hello world",
                }
            ),
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

        return {
            "statusCode": 200,
            "body": json.dumps(response_body),
        }

    # error
    else:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "bad request",
                }
            ),
        }
