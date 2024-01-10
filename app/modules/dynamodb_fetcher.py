import time
import boto3
import pandas as pd


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
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    table = dynamodb.Table(dynamodb_table_name)

    response = table.scan()

    df = pd.DataFrame(response["Items"])

    return df


# get data from dynamodb stream
def get_data_from_dynamodb_stream_while(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
) -> pd.DataFrame:
    """
    dynamodb streamからデータを取得する
    """

    # instance dynamodb
    dynamodb = boto3.client(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # DynamoDBストリームクライアントを作成
    dynamodbstreams = boto3.client(
        "dynamodbstreams",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # テーブルの詳細を取得
    table_description = dynamodb.describe_table(TableName=dynamodb_table_name)

    # 最新のストリームARNを取得
    latest_stream_arn = table_description["Table"]["LatestStreamArn"]

    # ストリームの詳細を取得
    stream_description = dynamodbstreams.describe_stream(StreamArn=latest_stream_arn)

    # 各シャードに対して
    for shard in stream_description["StreamDescription"]["Shards"]:
        # シャードイテレータを取得
        shard_iterator_response = dynamodbstreams.get_shard_iterator(
            StreamArn=latest_stream_arn,
            ShardId=shard["ShardId"],
            ShardIteratorType="TRIM_HORIZON",
        )

        # シャードイテレータ
        shard_iterator = shard_iterator_response["ShardIterator"]

        # レコードを取得
        start_time = time.time()
        timeout = 15
        while shard_iterator and time.time() - start_time < timeout:
            records_response = dynamodbstreams.get_records(ShardIterator=shard_iterator)

            if records_response["Records"] != []:
                return records_response["Records"][0]["dynamodb"]

            # 次のシャードイテレータを取得
            shard_iterator = records_response.get("NextShardIterator")
    return None


def get_data_from_dynamodb_stream(
    region_name: str,
    access_key_id: str,
    secret_access_key: str,
    dynamodb_table_name: str,
) -> dict | None:
    """
    dynamodb streamからデータを取得する
    """

    # instance dynamodb
    dynamodb = boto3.client(
        "dynamodb",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # DynamoDBストリームクライアントを作成
    dynamodbstreams = boto3.client(
        "dynamodbstreams",
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    # テーブルの詳細を取得
    table_description = dynamodb.describe_table(TableName=dynamodb_table_name)

    # 最新のストリームARNを取得
    latest_stream_arn = table_description["Table"]["LatestStreamArn"]

    # ストリームの詳細を取得
    stream_description = dynamodbstreams.describe_stream(StreamArn=latest_stream_arn)

    # 各シャードに対して
    for shard in stream_description["StreamDescription"]["Shards"]:
        # シャードイテレータを取得
        shard_iterator_response = dynamodbstreams.get_shard_iterator(
            StreamArn=latest_stream_arn,
            ShardId=shard["ShardId"],
            ShardIteratorType="TRIM_HORIZON",
        )

        # シャードイテレータ
        shard_iterator = shard_iterator_response["ShardIterator"]

        # レコードを取得
        records_response = dynamodbstreams.get_records(ShardIterator=shard_iterator)

        if records_response["Records"] != []:
            return records_response["Records"][0]["dynamodb"]
    return None
