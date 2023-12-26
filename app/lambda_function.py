import os
import json
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from dotenv import load_dotenv
from modules.model_operations import (
    init_and_retrain_model,
    make_predictions,
)
from modules.dynamodb_fetcher import (
    get_data_from_dynamodb,
    upload_dynamodb,
    delete_data_from_dynamodb,
    init_stock_table_dynamodb,
)
from modules.s3_fetcher import (
    download_model_from_s3,
    upload_model_to_s3,
)
from modules.dataframe_operations import (
    post_process_stock_data_from_dynamodb,
    get_unpredicted_data,
)


def handler(event, context):
    # read env
    load_dotenv(dotenv_path="../.env", override=True)
    tmp_dir: str = os.getenv("TMP_DIR")
    target_stock: str = os.getenv("TARGET_STOCK")
    stock_name: str = os.getenv("STOCK_NAME")
    period: str = os.getenv("PERIOD")
    interval: str = os.getenv("INTERVAL")
    df_col_order: list = os.getenv("DTAFRAME_COLUMNS_ORDER").split(",")
    model_num = int(os.getenv("MODEL_NUM"))
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME")
    dynamodb_stock_table_name = "spp_" + stock_name
    dynamodb_train_table_name = "spp_" + stock_name + "_trained"
    dynamo_pred_table_name = "spp_" + stock_name + "_pred"
    thread_pool_size = int(os.getenv("THREAD_POOL_SIZE"))

    # check env 
    if not all(
        [
            tmp_dir,
            target_stock,
            stock_name,
            period,
            interval,
            df_col_order,
            model_num,
            aws_region_name,
            aws_access_key_id,
            aws_secret_access_key,
            aws_s3_bucket_name,
        ]
    ):
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
    if event["handler"] == "init_stock_table_from_s3":
        # init dynamodb s3
        try:
            init_stock_table_dynamodb(
                tmp_dir,
                target_stock,
                stock_name,
                period,
                interval,
                df_col_order,
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
                aws_s3_bucket_name,
                thread_pool_size,
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
    elif event["handler"] == "init_stock_table_from_yfinance":
        # init dynamodb yfinance
        try:
            init_stock_table_dynamodb(
                tmp_dir,
                target_stock,
                stock_name,
                period,
                interval,
                df_col_order,
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
                aws_s3_bucket_name,
                thread_pool_size,
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

    # delete all item in pred table
    elif event["handler"] == "delete_pred_table_item":
        try:
            # pred tableからすべてのitemを取得
            df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
            )

            if df.empty:
                print("no item in pred table")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no item in pred table",
                        }
                    ),
                }

            # pred tableからすべてのitemを削除
            delete_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
                thread_pool_size,
                df,
            )
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to delete all item in pred table",
                    }
                ),
            }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to delete all item in pred table",
                }
            ),
        }

    # init model
    elif event["handler"] == "init_model":
        try:
            print("init model process is 3 steps")

            # download data from dynamodb
            print("step1: get all item from dynamodb stock table", end="")
            train_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
            )
            train_df = post_process_stock_data_from_dynamodb(train_df, df_col_order)
            print("...complete")

            print("step2: init and retrain model then dump to .pkl then upload s3")
            for i in tqdm(range(1, model_num + 1)):
                # init and retrain model
                model = init_and_retrain_model(train_df, i)

                # model to .pkl
                dump(
                    model,
                    f"{tmp_dir}/spp_{stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl",
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
            print("step2: complete")

            # get all data from dynamodb
            print("step3: init dynamodb trained table")
            df_to_delete = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_train_table_name,
            )

            if not df_to_delete.empty:
                print("train table is not empty")
                # delete all data from dynamodb
                delete_data_from_dynamodb(
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    dynamodb_train_table_name,
                    thread_pool_size,
                    df_to_delete,
                )

            # upload train data to dynamodb
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_train_table_name,
                thread_pool_size,
                train_df,
            )
            print("step3: complete")
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

    # update stock table

    # update predict table
    elif event["handler"] == "update_predict":
        try:
            # get all item from stock table
            stock_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
            )

            # post process get data from dynamodb
            stock_df = post_process_stock_data_from_dynamodb(stock_df, df_col_order)

            # get all item from pred table
            pred_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
            )

            # まだ予測していないデータを取得
            unpredicted_df = get_unpredicted_data(pred_df, stock_df)

            # unpredicted_dfが空の場合は、予測するデータがないので、終了
            if unpredicted_df.empty:
                print("no data to predict")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no data to predict",
                        }
                    ),
                }

            # model download from s3
            models = {}
            for i in range(1, model_num + 1):
                models[i] = download_model_from_s3(
                    tmp_dir,
                    stock_name,
                    i,
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_s3_bucket_name,
                )

            # 新たに予測したデータを格納するdfの作成
            update_pred_df = pd.DataFrame()
            update_pred_df["datetime"] = unpredicted_df[["datetime"]]

            # predict
            for key in models.keys():
                # predict
                y_predict = make_predictions(models[key], unpredicted_df)

                # concat predict to unpredicted_df
                update_pred_df[f"{key}_pred"] = y_predict

            # unpredicted_dfから最も大きいidを取得
            max_id = int(unpredicted_df["id"].max())

            # update_pred_dfにid列を追加
            update_pred_df["id"] = range(max_id + 1, max_id + 1 + len(update_pred_df))

            # update_pred_dfをs3のpred tableにアップロード
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
                thread_pool_size,
                update_pred_df,
            )

        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to update predict",
                    }
                ),
            }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to update predict",
                }
            ),
        }

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
    # handler({"handler": "delete_pred_table_item"}, None)
    handler({"handler": "init_model"}, None)
    # handler({"handler": "update_predict"}, None)
