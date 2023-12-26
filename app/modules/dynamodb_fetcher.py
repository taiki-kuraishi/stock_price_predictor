import boto3
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .yfinance_fetcher import get_all_from_yfinance
from .dataframe_operations import post_process_stock_data_from_dynamodb


def get_data_from_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
) -> pd.DataFrame:
    """
    get data from dynamodb
    dynamodbからデータを取得する
    """
    try:
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
    except Exception as e:
        print(e)
        raise Exception("fail to function on get_data_from_dynamodb")

    return df


def delete_data_from_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    thread_pool_size: int,
    df: pd.DataFrame,
) -> None:
    """
    delete data from dynamodb
    dynamodbからデータを削除する
    """
    try:
        dynamodb = boto3.client(
            "dynamodb",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        # if col datetime is type of datetime, convert to isoformat
        if df["datetime"].dtype == "datetime64[ns]":
            df["datetime"] = df["datetime"].dt.isoformat()

        for row in tqdm(df.to_dict("records")):
            dynamodb.delete_item(
                TableName=dynamodb_table_name,
                Key={
                    "id": {"N": str(row["id"])},
                    "datetime": {"S": row["datetime"]},
                },
            )

    except Exception as e:
        print(e)
        raise Exception("fail to function on delete_data_from_dynamodb")

    return None


def upload_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    thread_pool_size: int,
    df: pd.DataFrame,
) -> None:
    """
    upload data to dynamodb
    dynamodbに予測したデータをアップロードする
    """
    try:
        # instance of dynamodb
        dynamodb = boto3.client(
            "dynamodb",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        # if col datetime is type of datetime, convert to isoformat
        if df["datetime"].dtype == "datetime64[ns]":
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

        # col_list = df.columns.tolist()

        def put_item(row):
            dynamodb_item = {}

            # print(row)

            for key in row.keys():
                if key == "datetime":
                    dynamodb_item[str(key)] = {"S": row[key]}
                else:
                    dynamodb_item[str(key)] = {"N": str(row[key])}

            dynamodb.put_item(
                TableName=dynamodb_table_name,
                Item=dynamodb_item,
            )
            # print(f"{row.name+1}件目のデータを追加しました。")

        # multi thread
        with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
            list(tqdm(executor.map(put_item, df.to_dict("records")), total=df.shape[0]))

    except Exception as e:
        print(e)
        raise Exception("fail to function on upload_dynamodb")

    return None


def init_stock_table_dynamodb(
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
    thread_pool_size: int,
    data_source: str,
) -> None:
    """
    init dynamodb
    dynamodbを初期化する
    """
    try:
        print("init dynamodb process is 5 steps")

        # get all data from dynamodb
        print("step1: get all data from dynamodb", end="")
        df = get_data_from_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
        )
        df = post_process_stock_data_from_dynamodb(df, df_col_order)
        print("...complete")

        # delete all data from dynamodb
        print("step2: delete all data from dynamodb")
        delete_data_from_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
            thread_pool_size,
            df,
        )
        print("step2 : complete")

        if data_source == "s3":
            # download csv from s3
            print("step3: download csv from s3", end="")
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
            print("...complete")

            # read csv
            print("step4: read csv", end="")
            try:
                df = pd.read_csv(
                    f"{tmp_dir}/{file_name}", encoding="utf-8", index_col=None
                )
            except Exception as e:
                print(e)
                raise Exception("fail to read csv")
            print("...complete")

            # upload data to dynamodb
            print("step5: upload data to dynamodb")
            upload_dynamodb(
                region_name,
                access_key_id,
                secret_access_key,
                dynamodb_table_name,
                thread_pool_size,
                df,
            )
            print("step5: complete")
        elif data_source == "yfinance":
            # download data from yfinance
            print("step3: download data from yfinance")
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
            print("step3: complete")

            # upload data to dynamodb
            print("step4: upload data to dynamodb")
            upload_dynamodb(
                region_name,
                access_key_id,
                secret_access_key,
                dynamodb_table_name,
                thread_pool_size,
                df,
            )
            print("step4: complete")
        else:
            raise Exception("data_source is invalid")
    except Exception as e:
        print(e)
        raise Exception("fail to function on init_stock_table_dynamodb")

    return None


if __name__ == "__main__":
    """
    dynamodbのstockテーブルをs3に格納されたデータを使用して初期化する
    dynamodbのstockテーブルをyfinanceに格納されたデータを使用して初期化する
    """
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
    dynamodb_stock_table_name = "spp_" + stock_name
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME")

    print("do you want to init dynamodb? data source is s3 (y/n)")
    if input() == "y":
        # init dynamodb s3
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
            "s3",
        )
    else:
        print("init dynamodb canceled")

    print("do you want to init dynamodb? data source is yfinance (y/n)")
    if input() == "y":
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
            "yfinance",
        )
    else:
        print("init dynamodb canceled")
