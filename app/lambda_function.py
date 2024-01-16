import os
import pytz
import boto3
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from decimal import Decimal
from joblib import load, dump
from datetime import datetime, timedelta
from boto3.dynamodb.conditions import Key
from aws_lambda_context import LambdaContext
from sklearn.linear_model import PassiveAggressiveRegressor


def split_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    split target and features
    目的変数:xと説明変数:yに分割する
    """
    drop_columns = ["date", "time", "close"]
    y = df["close"]
    x = df.drop(drop_columns, axis=1)

    return x, y


def shift_dataFrame(df: pd.DataFrame, shift_rows: int) -> pd.DataFrame:
    """
    shift dataFrame
    shift_rows時間後の終値を予測するために、shift_rows行分だけデータをずらす
    """
    df_shifted = df.copy()
    df_shifted["close"] = df_shifted["close"].shift(-shift_rows)
    df_shifted.dropna(inplace=True, subset=["close"])

    return df_shifted


def post_process_stock_data_from_dynamodb(
    df: pd.DataFrame,
    df_col_order: list,
) -> pd.DataFrame:
    """
    post process stock data from dynamodb
    dynamodbから取得したデータは、カラムの順番や行の順番がバラバラなので、整形する
    """
    df = df.reindex(columns=df_col_order)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values(by=["datetime"])
    df = df.drop(["datetime"], axis=1)
    df = df.reset_index(drop=True)

    return df


def get_data_for_period_from_yfinance(
    target_stock: str,
    start_date: str,
    end_date: str,
    interval: str,
    df_col_order: list,
) -> pd.DataFrame:
    """
    get data from yfinance
    yfinanceからstart_dateからend_dateまでのデータを取得する
    """
    # get data from yahoo finance
    df = yf.download(
        tickers=target_stock, start=start_date, end=end_date, interval=interval
    )
    if df.empty:
        print("fail to get data from yahoo finance")
        raise Exception("fail to get data from yahoo finance. data is empty")
    else:
        print("success to get data from yahoo finance")
        print("rows : " + str(len(df)))

    # reset index and rename column
    df.reset_index(inplace=True, drop=False)

    # column name to lower
    df.columns = df.columns.str.lower()

    # rename column
    df.rename(columns={"date": "datetime"}, inplace=True)
    df.rename(columns={"adj close": "adj_close"}, inplace=True)

    # datetime to split date and time
    df["date"] = df["datetime"].apply(lambda x: x.date().isoformat())
    df["time"] = df["datetime"].apply(lambda x: x.time().isoformat())

    # drop col id,datetime
    df = df.drop(["id", "datetime"], axis=1)

    # sort columns
    df = df.reindex(columns=df_col_order)

    return df


def handler(event: dict, context: LambdaContext | None) -> dict:
    # read env
    timezone: str = os.environ["TIMEZONE"]
    tmp_dir: str = os.environ["TMP_DIR"]
    target_stock: str = os.environ["TARGET_STOCK"]
    stock_name: str = os.environ["STOCK_NAME"]
    period: str = os.environ["PERIOD"]
    interval: str = os.environ["INTERVAL"]
    df_col_order: list = os.environ["DTAFRAME_COLUMNS_ORDER"].split(",")
    model_num = int(os.environ["MODEL_NUM"])
    aws_region_name: str = os.environ["REGION_NAME"]
    aws_access_key_id: str = os.environ["ACCESS_KEY_ID"]
    aws_secret_access_key: str = os.environ["SECRET_ACCESS_KEY"]
    aws_s3_bucket_name: str = os.environ["AWS_S3_BUCKET_NAME"]
    dynamodb_stock_table_name: str = "stock_price_predictor_" + stock_name
    dynamodb_pred_table_name: str = dynamodb_stock_table_name + "_prediction"
    dynamo_limit_table_name: str = "stock_price_predictor_limit_value"

    JST = pytz.timezone(timezone)

    # aws instance
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    s3 = boto3.resource(
        "s3",
        region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    stock_table = dynamodb.Table(dynamodb_stock_table_name)
    prediction_table = dynamodb.Table(dynamodb_pred_table_name)
    limit_table = dynamodb.Table(dynamo_limit_table_name)

    # check handler
    if "handler" not in event:
        return {"message": "no handler"}

    # init limit table
    if event["handler"] == "init_limit_table":
        print("init limit table process is 1 steps")
        print("step1: all clear dynamodb limit table")
        limit_df_to_delete = pd.DataFrame(
            limit_table.scan()["Items"],
        )

        if not limit_df_to_delete.empty:
            only_key_df = limit_df_to_delete[["stock_id", "operation"]]

            for key in tqdm(only_key_df.to_dict("records")):
                limit_table.delete_item(Key=key)

        print("init limit table process is complete")
        return {
            "message": "success to init limit table",
        }

    # init train table
    elif event["handler"] == "init_stock_table":
        print("init stock table process is 5 steps")
        print("step1: all clear dynamodb stock table")
        stock_df_to_delete = pd.DataFrame(
            stock_table.scan()["Items"],
        )

        if not stock_df_to_delete.empty:
            only_key_df = stock_df_to_delete[["date", "time"]]

            for key in tqdm(only_key_df.to_dict("records")):
                stock_table.delete_item(Key=key)

        print("step2: download original stock data from s3")
        file_name = f"spp_{stock_name}_{period}_{interval}.csv"
        s3.Bucket(aws_s3_bucket_name).download_file(
            f"csv/{file_name}", f"{tmp_dir}/{file_name}"
        )
        original_df = pd.read_csv(
            f"{tmp_dir}/{file_name}", encoding="utf-8", index_col=None
        )

        # float型の値をDecimal型に変換
        original_df = original_df.apply(
            lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
        )

        print("step3: upload original stock data to dynamodb")
        with stock_table.batch_writer() as batch:
            for item in tqdm(original_df.to_dict("records")):
                batch.put_item(Item=item)

        print("step4: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "stock",
                "create_at": datetime.now(JST).isoformat(),
                "max": pd.to_datetime(
                    original_df.tail(1)["date"].values[0]
                    + " "
                    + original_df.tail(1)["time"].values[0]
                ).isoformat(),
            }
        )

        print("init stock table process is complete")
        return {
            "message": "success to init dynamodb",
        }

    # delete all item in pred table
    elif event["handler"] == "delete_pred_table_item":
        print("delete all item in pred table process is 3 steps")
        print("step1: get all item from dynamodb pred table")
        prediction_df_to_delete = pd.DataFrame(prediction_table.scan()["Items"])

        if not prediction_df_to_delete.empty:
            print("step2: delete all item from dynamodb pred table")
            only_key_prediction_df = prediction_df_to_delete[["date", "time"]]

            for key in tqdm(only_key_prediction_df.to_dict("records")):
                prediction_table.delete_item(Key=key)

            print("step2: complete")
            print("delete all item in pred table process is complete")

        print("step3: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "prediction",
                "create_at": datetime.now(JST).isoformat(),
                "max": "0",
            }
        )

        print("delete all item in pred table process is complete")
        return {
            "message": "success to delete all item in pred table",
        }

    # init model
    elif event["handler"] == "init_model":
        print("init model process is 3 steps")
        print("step1: get all item from dynamodb stock table", end="")
        train_df = pd.DataFrame(stock_table.scan()["Items"])
        train_df = post_process_stock_data_from_dynamodb(train_df, df_col_order)
        print("...complete")

        print("step2: init and retrain model then dump to .pkl then upload s3")
        for i in tqdm(range(1, model_num + 1)):
            model_file_name = (
                f"spp_{stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl"
            )
            model_local_path = f"{tmp_dir}/{model_file_name}"

            shifted_train_df = shift_dataFrame(train_df, i)
            x, y = split_target_and_features(shifted_train_df)
            model = PassiveAggressiveRegressor()
            model.fit(x, y)
            dump(model, model_local_path)
            s3.Bucket(aws_s3_bucket_name).upload_file(
                model_local_path, f"models/{model_file_name}"
            )
        print("step2: complete")

        print("step3: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "train",
                "create_at": datetime.now(JST).isoformat(),
                "max": pd.to_datetime(
                    train_df.tail(1)["date"].values[0]
                    + " "
                    + train_df.tail()["time"].values[0]
                ).isoformat(),
            }
        )

        print("init model process is complete")
        return {
            "message": "success to init model",
        }

    # update stock table
    elif event["handler"] == "update_stock_table":
        print("update stock table process is 3 steps")
        print("step1: get latest date from limit table")
        latest_stock_date = limit_table.get_item(
            Key={"stock_id": target_stock, "operation": "stock"}
        )["Item"]

        stock_max_datetime = pd.to_datetime(latest_stock_date["max"])
        start_date = stock_max_datetime.strftime("%Y-%m-%d")
        end_date = (datetime.now(JST) + timedelta(days=1)).strftime("%Y-%m-%d")
        print("\tstart date : " + start_date)
        print("\tend date : " + end_date)

        print("step2: get latest stock data")
        yfinance_response_df = yf.download(
            tickers=target_stock, start=start_date, end=end_date, interval=interval
        )
        if yfinance_response_df.empty:
            print("fail to get data from yahoo finance")
            raise Exception("fail to get data from yahoo finance. data is empty")

        yfinance_response_df.reset_index(inplace=True, drop=False)
        yfinance_response_df.columns = yfinance_response_df.columns.str.lower()

        # rename column
        yfinance_response_df.rename(columns={"date": "datetime"}, inplace=True)
        yfinance_response_df.rename(columns={"adj close": "adj_close"}, inplace=True)

        yfinance_response_df["datetime"] = pd.to_datetime(
            yfinance_response_df["datetime"]
        )
        # convert timezone
        yfinance_response_df["datetime"] = yfinance_response_df[
            "datetime"
        ].dt.tz_convert(JST)

        # delete timezone
        yfinance_response_df["datetime"] = yfinance_response_df[
            "datetime"
        ].dt.tz_localize(None)

        # すでに格納されているデータは除外
        yfinance_response_df = yfinance_response_df[
            yfinance_response_df["datetime"] > stock_max_datetime
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

        if yfinance_response_df.empty:
            print("no data to update")
            return {
                "message": "no data to update",
            }

        print("step3: upload latest stock data to dynamodb")
        with stock_table.batch_writer() as batch:
            for item in tqdm(yfinance_response_df.to_dict("records")):
                batch.put_item(Item=item)

        print("step4: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "stock",
                "create_at": datetime.now(JST).isoformat(),
                "max": pd.to_datetime(
                    yfinance_response_df.tail(1)["date"].values[0]
                    + " "
                    + yfinance_response_df.tail(1)["time"].values[0]
                ).isoformat(),
            }
        )

        print("update stock table process is complete")
        return {
            "message": "success to update stock table",
        }

    # update predict table
    elif event["handler"] == "update_predict":
        print("update predict process is 3 steps")
        print("step1: get latest date from limit table")
        limit_prediction_date = limit_table.get_item(
            Key={"stock_id": target_stock, "operation": "prediction"}
        )["Item"]

        if limit_prediction_date["max"] == "0":
            unpredicted_df = pd.DataFrame(stock_table.scan()["Items"])
        else:
            prediction_max_datetime = pd.to_datetime(limit_prediction_date["max"])
            prediction_max_date = prediction_max_datetime.strftime("%Y-%m-%d")
            prediction_max_time = prediction_max_datetime.strftime("%H:%M:%S")

            limit_stock_date = limit_table.get_item(
                Key={"stock_id": target_stock, "operation": "stock"}
            )["Item"]
            predict_end_date = limit_stock_date["max"]
            predict_start_date = prediction_max_date
            print("\tpredict start date: " + predict_start_date)
            print("\tpredict end date: " + predict_end_date)

            unpredicted_df = pd.DataFrame()

            while predict_start_date <= predict_end_date:
                day_of_prediction_df = pd.DataFrame(
                    stock_table.query(
                        KeyConditionExpression=Key("date").eq(predict_start_date)
                    )["Items"]
                )
                # すでに予測済みのデータは除外
                if predict_start_date == prediction_max_date:
                    day_of_prediction_df = day_of_prediction_df[
                        day_of_prediction_df["time"] > prediction_max_time
                    ]
                unpredicted_df = pd.concat([unpredicted_df, day_of_prediction_df])
                predict_start_date = (
                    datetime.strptime(predict_start_date, "%Y-%m-%d")
                    + timedelta(days=1)
                ).strftime("%Y-%m-%d")

        if unpredicted_df.empty:
            print("no data to predict")
            return {
                "message": "no data to predict",
            }

        # カラムのソートと日付でソート
        unpredicted_df = post_process_stock_data_from_dynamodb(
            unpredicted_df, df_col_order
        )

        # 予測するデータを目的変数と説明変数に分割
        x, _ = split_target_and_features(unpredicted_df)

        # 新たに予測したデータを格納するdfの作成
        update_prediction_df = pd.DataFrame()
        update_prediction_df["date"] = unpredicted_df[["date"]]
        update_prediction_df["time"] = unpredicted_df[["time"]]

        # model download from s3
        print("step2: download model from s3")
        for i in tqdm(range(1, model_num + 1)):
            model_file_name = (
                f"spp_{stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl"
            )
            model_local_path = f"{tmp_dir}/{model_file_name}"
            s3.Bucket(aws_s3_bucket_name).download_file(
                f"models/{model_file_name}", model_local_path
            )
            model = load(model_local_path)
            update_prediction_df[f"{i}_pred"] = model.predict(x)

        # convert float to decimal
        update_prediction_df = update_prediction_df.apply(
            lambda x: x.map(lambda y: Decimal(str(y)) if isinstance(y, float) else y)
        )

        # create_at column
        update_prediction_df["create_at"] = datetime.now(JST).isoformat()

        print("To update dataFrame : \n")
        print(update_prediction_df)

        print("step5: upload predict data to dynamodb")
        with prediction_table.batch_writer() as batch:
            for item in tqdm(update_prediction_df.to_dict("records")):
                batch.put_item(Item=item)

        print("step6: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "prediction",
                "create_at": datetime.now(JST).isoformat(),
                "max": pd.to_datetime(
                    update_prediction_df.tail(1)["date"].values[0]
                    + " "
                    + update_prediction_df.tail(1)["time"].values[0]
                ).isoformat(),
            }
        )

        print("update predict process is complete")
        return {
            "message": "success to update predict",
        }

    # update_and_predict_tables
    elif event["handler"] == "update_and_predict_tables":
        print("update_and_predict_tables process")
        update_stock_table_res = handler({"handler": "update_stock_table"}, context)
        update_stock_table_res_message: str = update_stock_table_res["message"]

        update_predict_res = handler({"handler": "update_predict"}, context)
        update_predict_res_message: str = update_predict_res["message"]

        print("update_and_predict_tables process is complete")

        return {
            "message": "success to update_and_predict_tables",
            "update_stock_table": update_stock_table_res_message,
            "update_predict": update_predict_res_message,
        }

    # update model
    elif event["handler"] == "update_model":
        print("update model process is 3 steps")
        print("step1: get all item from dynamodb stock table")
        limit_train_date = limit_table.get_item(
            Key={"stock_id": target_stock, "operation": "train"}
        )["Item"]
        # train_max_dateに1日足した日付から予測を開始する
        train_max_date = (
            pd.to_datetime(limit_train_date["max"]) + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        limit_stock_date = limit_table.get_item(
            Key={"stock_id": target_stock, "operation": "stock"}
        )["Item"]

        train_end_date = pd.to_datetime(limit_stock_date["max"]).strftime("%Y-%m-%d")
        print("\ttrain start date: " + train_max_date)
        print("\ttrain end date: " + train_end_date)

        untrained_df = pd.DataFrame()
        date = train_max_date
        while date <= train_end_date:
            day_of_train_df = pd.DataFrame(
                stock_table.query(KeyConditionExpression=Key("date").eq(date))["Items"]
            )
            untrained_df = pd.concat([untrained_df, day_of_train_df])
            date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

        if untrained_df.empty:
            print("no data to train")
            return {
                "message": "no data to train",
            }

        # カラムのソートと日付でソート
        untrained_df = post_process_stock_data_from_dynamodb(untrained_df, df_col_order)

        print("untrained_df : \n")
        print(untrained_df)

        # model download from s3
        print("step2: download model and incremental learning and upload s3")
        for i in tqdm(range(1, model_num + 1)):
            model_file_name = (
                f"spp_{stock_name}_{str(i)}h_PassiveAggressiveRegressor.pkl"
            )
            model_local_path = f"{tmp_dir}/{model_file_name}"
            s3.Bucket(aws_s3_bucket_name).download_file(
                f"models/{model_file_name}", model_local_path
            )
            train_df = shift_dataFrame(untrained_df, i)
            x, y = split_target_and_features(train_df)
            model = load(model_local_path)
            model.partial_fit(x, y)
            dump(model, model_local_path)
            s3.Bucket(aws_s3_bucket_name).upload_file(
                model_local_path, f"models/{model_file_name}"
            )

        print("step4: upload limit value to dynamodb")
        limit_table.put_item(
            Item={
                "stock_id": target_stock,
                "operation": "train",
                "create_at": datetime.now(JST).isoformat(),
                "max": train_end_date,
            }
        )

        print("update model process is complete")
        return {
            "message": "success to update model",
        }

    else:
        return {
            "message": "wrong handler",
        }
