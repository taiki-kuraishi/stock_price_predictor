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
    # instance of dynamodb
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # instance of table
    table = dynamodb.Table(dynamodb_table_name)

    # get all data from dynamodb
    response = table.scan()

    # convert to dataframe
    df = pd.DataFrame(response["Items"])

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

    for row in tqdm(df.to_dict("records")):
        dynamodb.delete_item(
            TableName=dynamodb_table_name,
            Key={
                "id": {"N": str(row["id"])},
                "datetime": {"S": row["datetime"]},
            },
        )

    return None


def upload_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    df: pd.DataFrame,
    thread_pool_size: int = 1,
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
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # col_list = df.columns.tolist()

    def put_item(row: dict) -> None:
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
        df,
    )
    print("step2:complete")

    if data_source == "s3":
        # download csv from s3
        print("step3: download csv from s3", end="")

        # instance s3 client
        s3 = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        # file name
        file_name = f"spp_{stock_name}_{period}_{interval}.csv"

        # download csv from s3
        s3.download_file(
            s3_bucket_name, f"csv/{file_name}", f"{tmp_dir}/{file_name}"
        )
        print("...complete")

        # read csv
        print("step4: read csv", end="")
        df = pd.read_csv(f"{tmp_dir}/{file_name}", encoding="utf-8", index_col=None)
        print("...complete")

        # upload data to dynamodb
        print("step5: upload data to dynamodb")
        upload_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
            df,
            thread_pool_size,
        )
        print("step5: complete")
    elif data_source == "yfinance":
        # download data from yfinance
        print("step3: download data from yfinance")
        df = get_all_from_yfinance(
            target_stock,
            period,
            interval,
            df_col_order,
        )
        print("step3: complete")

        # upload data to dynamodb
        print("step4: upload data to dynamodb")
        upload_dynamodb(
            region_name,
            access_key_id,
            secret_access_key,
            dynamodb_table_name,
            df,
            thread_pool_size,
        )
        print("step4: complete")
    else:
        raise Exception("data_source is invalid")

    return None
