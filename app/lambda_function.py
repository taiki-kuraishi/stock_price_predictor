import os
import json
import pandas as pd
from tqdm import tqdm
from joblib import dump
from dotenv import load_dotenv
from aws_lambda_context import LambdaContext
from modules.dataframe_operations import (
    post_process_stock_data_from_dynamodb,
    get_latest_stock_data,
    get_rows_not_in_other_df,
)
from modules.dynamodb_fetcher import (
    get_data_from_dynamodb,
    upload_dynamodb,
    delete_data_from_dynamodb,
    init_stock_table_dynamodb,
)
from modules.model_operations import (
    init_and_retrain_model,
    incremental_learning,
    make_predictions,
)
from modules.s3_fetcher import (
    download_model_from_s3,
    upload_model_to_s3,
)


def handler(event: dict, context: LambdaContext | None) -> dict:
    # read env
    try:
        tmp_dir: str = os.environ["TMP_DIR"]
        target_stock: str = os.environ["TARGET_STOCK"]
        stock_name: str = os.environ["STOCK_NAME"]
        period: str = os.environ["PERIOD"]
        interval: str = os.environ["INTERVAL"]
        df_col_order: list = os.environ["DTAFRAME_COLUMNS_ORDER"].split(",")
        model_num = int(os.environ["MODEL_NUM"])
        aws_region_name: str = os.environ["AWS_REGION_NAME"]
        aws_access_key_id: str = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key: str = os.environ["AWS_SECRET_ACCESS_KEY"]
        aws_s3_bucket_name: str = os.environ["AWS_S3_BUCKET_NAME"]
        dynamodb_stock_table_name = "spp_" + stock_name
        dynamodb_train_table_name = "spp_" + stock_name + "_trained"
        dynamo_pred_table_name = "spp_" + stock_name + "_pred"
        thread_pool_size = int(os.environ["THREAD_POOL_SIZE"])
    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "fail to read env",
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
            print("delete all item in pred table process is 2 steps")
            # pred tableからすべてのitemを取得
            print("step1: get all item from dynamodb pred table", end="")
            df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
            )
            df = post_process_stock_data_from_dynamodb(df, df_col_order)
            print("...complete")

            if df.empty:
                print("no item in pred table")
                print("delete all item in pred table process is complete")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no item in pred table",
                        }
                    ),
                }

            # pred tableからすべてのitemを削除
            print("step2: delete all item from dynamodb pred table")
            delete_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
                df,
            )
            print("step2: complete")
            print("delete all item in pred table process is complete")
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
            print("step3: complete")

            if not df_to_delete.empty:
                print("train table is not empty")
                print("step3.5 : delete all item from dynamodb train table")
                # delete all data from dynamodb
                delete_data_from_dynamodb(
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    dynamodb_train_table_name,
                    df_to_delete,
                )
                print("step3.5: complete")

            # train_dfのid, datetime列を抽出
            print("step4: upload trained data to dynamodb")
            to_upload_train_df = train_df[["id", "datetime"]]
            # upload train data to dynamodb
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_train_table_name,
                to_upload_train_df,
            )
            print("step4: complete")
            print("init model process is complete")
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
    elif event["handler"] == "update_stock_table":
        try:
            print("update stock table process is 2 steps")
            # 現在のstock tableのデータを取得
            print("step1: get all item from dynamodb stock table", end="")
            old_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
            )
            old_df = post_process_stock_data_from_dynamodb(old_df, df_col_order)
            print("...complete")

            # get latest stock data
            print("step2: get latest stock data")
            latest_stock_df = get_latest_stock_data(
                target_stock, interval, df_col_order, old_df
            )

            if latest_stock_df is None:
                print("no data to update")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no data to update",
                        }
                    ),
                }
            print("...complete")

            # latest_stock_dfをs3のstock tableにアップロード
            print("step3: upload latest stock data to dynamodb")
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
                latest_stock_df,
                thread_pool_size,
            )
            print("step3: complete")
            print("update stock table process is complete")

        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to update stock table",
                    }
                ),
            }

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to update stock table",
                }
            ),
        }

    # update predict table
    elif event["handler"] == "update_predict":
        try:
            print("update predict process is 3 steps")

            # get all item from stock table
            print("step1: get all item from dynamodb stock table", end="")
            stock_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
            )
            stock_df = post_process_stock_data_from_dynamodb(stock_df, df_col_order)
            print("...complete")

            # get all item from pred table
            print("step2: get all item from dynamodb pred table", end="")
            pred_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
            )
            pred_df = post_process_stock_data_from_dynamodb(pred_df, df_col_order)
            print("...complete")

            # まだ予測していないデータを取得
            unpredicted_df = get_rows_not_in_other_df(pred_df, stock_df)

            # unpredicted_dfが空の場合は、予測するデータがないので、終了
            if unpredicted_df.empty:
                print("no data to predict")
                print("update predict process is complete")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no data to predict",
                        }
                    ),
                }

            # model download from s3
            print("step3: download model from s3")
            models = {}
            for i in tqdm(range(1, model_num + 1)):
                models[i] = download_model_from_s3(
                    tmp_dir,
                    stock_name,
                    i,
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_s3_bucket_name,
                )
            print("step3: complete")

            # 新たに予測したデータを格納するdfの作成
            update_pred_df = pd.DataFrame()
            update_pred_df["datetime"] = unpredicted_df[["datetime"]]

            # predict
            print("step4: predict")
            for key in tqdm(models.keys()):
                # predict
                y_predict = make_predictions(models[key], unpredicted_df)

                # concat predict to unpredicted_df
                update_pred_df[f"{key}_pred"] = y_predict

            # unpredicted_dfから最も大きいidを取得
            max_id = int(unpredicted_df["id"].max())

            # update_pred_dfにid列を追加
            update_pred_df["id"] = range(max_id + 1, max_id + 1 + len(update_pred_df))
            print("step4: complete")

            # update_pred_dfをs3のpred tableにアップロード
            print("step5: upload predict data to dynamodb")
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamo_pred_table_name,
                update_pred_df,
                thread_pool_size,
            )
            print("step5: complete")
            print("update predict process is complete")
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
    elif event["handler"] == "update_model":
        try:
            print("update model process is 3 steps")

            # get all item from stock table
            print("step1: get all item from dynamodb stock table", end="")
            stock_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_stock_table_name,
            )
            stock_df = post_process_stock_data_from_dynamodb(stock_df, df_col_order)
            print("...complete")

            # get all item from train table
            print("step2: get all item from dynamodb train table", end="")
            train_df = get_data_from_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_train_table_name,
            )
            train_df = post_process_stock_data_from_dynamodb(train_df, df_col_order)
            print("...complete")

            # 未学習のデータを抽出する
            untrained_df = get_rows_not_in_other_df(train_df, stock_df)
            untrained_length = len(untrained_df)
            print(f"untrained data num : {untrained_length}")

            # untrained_dfが空の場合は、学習するデータがないので、終了
            if untrained_df.empty:
                print("no data to train")
                print("update model process is complete")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "no data to train",
                        }
                    ),
                }

            # model download from s3
            print("step3: download model from s3")
            models = {}
            for i in tqdm(range(1, model_num + 1)):
                models[i] = download_model_from_s3(
                    tmp_dir,
                    stock_name,
                    i,
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_s3_bucket_name,
                )
            print("step3: complete")

            # incrementally train
            print("step4: train and convert to .pkl then upload s3")
            for key in tqdm(models.keys()):
                model = incremental_learning(
                    models[key], stock_df.tail(untrained_length + key), key
                )
                dump(
                    model,
                    f"{tmp_dir}/spp_{stock_name}_{str(key)}h_PassiveAggressiveRegressor.pkl",
                )
                upload_model_to_s3(
                    tmp_dir,
                    stock_name,
                    key,
                    model,
                    aws_region_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_s3_bucket_name,
                )
            print("step4: complete")

            # upload train data to dynamodb
            print("step5: upload train data to dynamodb")
            upload_dynamodb(
                aws_region_name,
                aws_access_key_id,
                aws_secret_access_key,
                dynamodb_train_table_name,
                untrained_df,
                thread_pool_size,
            )
            print("step5: complete")
            print("update model process is complete")
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to update model",
                    }
                ),
            }

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to update model",
                }
            ),
        }

    else:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "wrong handler",
                }
            ),
        }


if __name__ == "__main__":
    load_dotenv(dotenv_path="../.env", override=True)
    handler({"handler": ""}, None)
    handler({"handler": "init_stock_table_from_s3"}, None)
    # handler({"handler": "init_stock_table_from_yfinance"}, None)
    handler({"handler": "delete_pred_table_item"}, None)
    handler({"handler": "init_model"}, None)
    handler({"handler": "update_predict"}, None)
    handler({"handler": "update_stock_table"}, None)
    handler({"handler": "update_model"}, None)
