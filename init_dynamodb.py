import os
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
data_dir = os.environ["DATA_DIR"]
models_dir = os.environ["MODEL_DIR"]
target_stock = os.environ["TARGET_STOCK"]
stock_name = os.environ["STOCK_NAME"]
period = os.environ["PERIOD"]
interval = os.environ["INTERVAL"]
predict_horizon = 1
aws_region_name = os.environ["AWS_REGION_NAME"]
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
# aws_s3_bucket_name = os.environ["AWS_S3_BUCKET_NAME"]
aws_dynamodb_table_name = os.environ["AWS_DYNAMODB_TABLE_NAME"]


dynamodb = boto3.client(
    "dynamodb",
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# read csv
df = pd.read_csv(
    f"{aws_dynamodb_table_name}.csv",
    encoding="utf-8",
    index_col=0,
)
# index reset
df = df.reset_index(drop=False)
print(df.head())

# col rename to lowercase
df.columns = df.columns.str.lower()
print(df.head())

# col rename
df = df.rename(columns={"index": "id"})
df = df.rename(columns={"adj close": "adj_close"})
print(df.head())

df_col = df.columns.tolist()
print(df_col)

for i in range(len(df)):
    row = df.iloc[i].to_dict()
    dynamodb_item = {
        "id": {"N": str(row["id"])},
        "datetime": {"S": row["datetime"]},
        "open": {"N": str(row["open"])},
        "high": {"N": str(row["high"])},
        "low": {"N": str(row["low"])},
        "close": {"N": str(row["close"])},
        "adj_close": {"N": str(row["adj_close"])},
        "volume": {"N": str(row["volume"])},
    }  # Convert each item to a dictionary
    dynamodb.put_item(
        TableName=aws_dynamodb_table_name,
        Item=dynamodb_item,
    )
    print(f"{i+1}件目のデータを追加しました。")
