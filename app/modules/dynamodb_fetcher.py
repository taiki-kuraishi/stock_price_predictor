import boto3
import pandas as pd


def get_data_from_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    df_col_order: list,
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

    response = table.scan()

    df = pd.DataFrame(response["Items"])

    # sort columns
    df = df.reindex(columns=df_col_order)

    # sort id
    df = df.sort_values("id", ascending=True)

    # reset index
    df = df.reset_index(drop=True)

    return df


def upload_data_to_dynamodb(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    upload data to dynamodb
    dynamodbにデータをアップロードする
    """
    dynamodb = boto3.client(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        dynamodb_item = {
            "id": {"N": str(row["id"])},
            "datetime": {"S": row["datetime"].isoformat()},
            "open": {"N": str(row["open"])},
            "high": {"N": str(row["high"])},
            "low": {"N": str(row["low"])},
            "close": {"N": str(row["close"])},
            "adj_close": {"N": str(row["adj_close"])},
            "volume": {"N": str(row["volume"])},
        }
        dynamodb.put_item(
            TableName=dynamodb_table_name,
            Item=dynamodb_item,
        )
        print(f"{i+1}件目のデータを追加しました。")

    return
