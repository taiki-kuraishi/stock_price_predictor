import boto3
import pandas as pd
from yfinance_fetcher import get_all_from_yfinance


def get_data_from_dynamodb(
    df_col_order: list,
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
) -> pd.DataFrame:
    """
    get data from dynamodb
    dynamodbからデータを取得する
    """
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    table = dynamodb.Table(dynamodb_table_name)

    try:
        response = table.scan()
    except Exception as e:
        print(e)
        raise Exception("fail to get data from dynamodb")

    df = pd.DataFrame(response["Items"])

    # sort columns
    df = df.reindex(columns=df_col_order)

    # sort id
    df = df.sort_values("id", ascending=True)

    # reset index
    df = df.reset_index(drop=True)

    return df

def delete_data_from_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    delete data from dynamodb
    dynamodbからデータを削除する
    """
    dynamodb = boto3.client(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # if col datetime is type of datetime, convert to isoformat
    if df["datetime"].dtype == "datetime64[ns]":
        df["datetime"] = df["datetime"].dt.isoformat()
    try:
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            dynamodb.delete_item(
                TableName=dynamodb_table_name,
                Key={
                    "id": {"N": str(row["id"])},
                    "datetime": {"S": row["datetime"]},
                },
            )
            print(f"{i+1}件目のデータを削除しました。")
    except Exception as e:
        print(e)
        raise Exception("fail to delete data from dynamodb")

    return None


def upload_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    upload data to dynamodb
    dynamodbに予測したデータをアップロードする
    """
    # instance of dynamodb
    dynamodb = boto3.client(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # if col datetime is type of datetime, convert to isoformat
    if df["datetime"].dtype == "datetime64[ns]":
        df["datetime"] = df["datetime"].dt.isoformat()

    col_list = df.columns.tolist()

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        dynamodb_item = {}

        for col in col_list:
            if col == "datetime":
                dynamodb_item[str(col)] = {"S": row[col]}
            else:
                dynamodb_item[str(col)] = {"N": str(row[col])}

        dynamodb.put_item(
            TableName=dynamodb_table_name,
            Item=dynamodb_item,
        )
        print(f"{i+1}件目のデータを追加しました。")

    return None


def init_dynamodb(
    tmp_dir: str,
    target_stock: str,
    stock_name: str,
    period: str,
    interval: str,
    df_col_order: list,
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    s3_bucket_name: str,
    data_source: str,
) -> None:
    """
    init dynamodb
    dynamodbを初期化する
    """

    # get all data from dynamodb
    df = get_data_from_dynamodb(
        df_col_order,
        region_name,
        access_key_id,
        secret_access_key,
        dynamodb_table_name,
    )

    # delete all data from dynamodb
    delete_data_from_dynamodb(
        region_name,
        access_key_id,
        secret_access_key,
        dynamodb_table_name,
        df,
    )

    if data_source == "s3":
        # download csv from s3
        s3 = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        file_name = f"spp_{stock_name}_{period}_{interval}.csv"
        try:
            s3.download_file(
                s3_bucket_name, f"csv/{file_name}", f"{tmp_dir}/{file_name}"
            )
        except Exception as e:
            print(e)
            raise Exception("fail to download csv from s3")

        # read csv
        try:
            df = pd.read_csv(f"{tmp_dir}/{file_name}", encoding="utf-8", index_col=None)
        except Exception as e:
            print(e)
            raise Exception("fail to read csv")

        # upload data to dynamodb
        upload_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
            df,
        )
    elif data_source == "yfinance":
        # download data from yfinance
        try:
            df = get_all_from_yfinance(
                target_stock,
                period,
                interval,
                df_col_order,
            )
        except Exception as e:
            print(e)
            raise Exception("fail to download data from yfinance")

        # upload data to dynamodb
        upload_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
            df,
        )
    else:
        raise Exception("data_source is invalid")

    print("init_dynamodb complete")
    return None


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../../.env", override=True)
    tmp_dir: str = "../../tmp"
    target_stock: str = os.getenv("TARGET_STOCK")
    stock_name: str = os.getenv("STOCK_NAME")
    period: str = os.getenv("PERIOD")
    interval: str = os.getenv("INTERVAL")
    df_col_order: list = os.getenv("DTAFRAME_COLUMNS_ORDER").split(",")
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_dynamodb_train_table_name: str = os.getenv("AWS_DYNAMODB_TRAIN_TABLE_NAME")
    aws_dynamodb_pred_table_name: str = os.getenv("AWS_DYNAMODB_PREDICTION_TABLE_NAME")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME")

    print("do you want to init dynamodb? data source is s3 (y/n)")
    if input() == "y":
        # init dynamodb s3
        init_dynamodb(
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
    else:
        print("init dynamodb canceled")

    print("do you want to init dynamodb? data source is yfinance (y/n)")
    if input() == "y":
        init_dynamodb(
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
    else:
        print("init dynamodb canceled")