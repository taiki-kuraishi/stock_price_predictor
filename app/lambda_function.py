import os
import json
from joblib import dump, load
from dotenv import load_dotenv
from modules.model_operations import (
    init_and_retrain_model,
    make_predictions,
    get_updated_dataFrame,
)
from modules.dynamodb_fetcher import (
    get_data_from_dynamodb,
    upload_dynamodb,
    init_train_table_dynamodb,
)
from modules.s3_fetcher import (
    download_model_from_s3,
    upload_model_to_s3,
)
from modules.dataframe_operations import post_process_train_data_from_dynamodb


def handler(event, context):
    # read env
    load_dotenv(dotenv_path="../.env", override=True)
    tmp_dir: str = os.getenv("TMP_DIR")
    target_stock: str = os.getenv("TARGET_STOCK")
    stock_name: str = os.getenv("STOCK_NAME")
    period: str = os.getenv("PERIOD")
    interval: str = os.getenv("INTERVAL")
    df_col_order: list = os.getenv("DTAFRAME_COLUMNS_ORDER").split(",")
    model_num: int = os.getenv("MODEL_NUM")
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME")
    aws_dynamodb_train_table_name: str = os.getenv("AWS_DYNAMODB_TRAIN_TABLE_NAME")
    aws_dynamo_prediction_table_name: str = os.getenv(
        "AWS_DYNAMODB_PREDICTION_TABLE_NAME"
    )

    # check env
    if not all([tmp_dir, target_stock, stock_name, period, interval]):
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "fail to read env. not all env are set",
                }
            ),
        }

    # check handler
    if "handler" not in event:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "handler is not set",
                }
            ),
        }

    # init train table
    if event["handler"] == "init_train_table_from_s3":
        # init dynamodb s3
        try:
            init_train_table_dynamodb(
                tmp_dir,
                target_stock,
                stock_name,
                period,
                interval,
                df_col_order,
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                aws_dynamodb_train_table_name,
                aws_s3_bucket_name,
                "s3",
            )
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to init dynamodb",
                    }
                ),
            }

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to init dynamodb",
                }
            ),
        }
    elif event["handler"] == "init_train_table_from_yfinance":
        # init dynamodb yfinance
        try:
            init_train_table_dynamodb(
                tmp_dir,
                target_stock,
                stock_name,
                period,
                interval,
                df_col_order,
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                aws_dynamodb_train_table_name,
                aws_s3_bucket_name,
                "yfinance",
            )
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to init dynamodb",
                    }
                ),
            }

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to init dynamodb",
                }
            ),
        }

    # init pred table
    elif event["handler"] == "delete_pred_table_item":
        ...
    # init model
    elif event["handler"] == "init_model":
        try:
            # download data from dynamodb
            df = get_data_from_dynamodb(
                df_col_order,
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                aws_dynamodb_train_table_name,
            )

            df = post_process_train_data_from_dynamodb(df, df_col_order)

            for i in range(1, model_num + 1):
                # init and retrain model
                model = init_and_retrain_model(df, i)

                # model to .pkl
                dump(
                    model,
                    f"spp_{stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl",
                )

                # upload model to s3
                upload_model_to_s3(
                    tmp_dir,
                    stock_name,
                    i,
                    model,
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_s3_bucket_name,
                )
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to init model",
                    }
                ),
            }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to init model",
                }
            ),
        }

    # update predict
    # elif event["handler"] == "update_predict":
    # update model
    else:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "handler is not set",
                }
            ),
        }


if __name__ == "__main__":
    handler({"handler": "init"}, None)
    handler({"handler": "update"}, None)
