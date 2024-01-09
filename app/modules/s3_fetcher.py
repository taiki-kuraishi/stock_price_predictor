import boto3
from joblib import load, dump
from sklearn.linear_model import PassiveAggressiveRegressor


def download_model_from_s3(
    tmp_dir: str,
    stock_name: str,
    predict_horizon: int,
    aws_region_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_s3_bucket_name: str,
) -> PassiveAggressiveRegressor:
    """
    download model data from s3
    """
    # model file name
    model_file_name = (
        f"spp_{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl"
    )
    model_local_path = f"{tmp_dir}/{model_file_name}"

    # instance s3 client
    s3 = boto3.client(
        "s3",
        region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # download model from s3
    s3.download_file(aws_s3_bucket_name, f"models/{model_file_name}", model_local_path)

    # load model
    model = load(model_local_path)

    return model


def upload_model_to_s3(
    tmp_dir: str,
    stock_name: str,
    predict_horizon: int,
    model: PassiveAggressiveRegressor,
    aws_region_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_s3_bucket_name: str,
) -> None:
    """
    upload model data to s3
    モデルを受け取って.pklにしてs3にアップロードする
    """
    # model file name
    model_file_name = (
        f"spp_{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl"
    )
    model_local_path = f"{tmp_dir}/{model_file_name}"

    # model to .pkl
    dump(model, model_local_path)

    # instance s3 client
    s3 = boto3.client(
        "s3",
        region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # upload model to s3
    s3.upload_file(model_local_path, aws_s3_bucket_name, f"models/{model_file_name}")

    return None
