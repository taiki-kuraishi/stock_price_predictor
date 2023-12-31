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

