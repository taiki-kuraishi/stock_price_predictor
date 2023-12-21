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
    try:
        # model file name
        model_file_name = (
            f"spp_{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl"
        )
        model_local_path = f"{tmp_dir}/{model_file_name}"

        s3 = boto3.client(
            "s3",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        try:
            s3.download_file(
                aws_s3_bucket_name, f"models/{model_file_name}", model_local_path
            )
        except Exception as e:
            print(e)
            raise Exception("fail to download model from s3 on download_model_from_s3")

        try:
            model = load(model_local_path)
        except Exception as e:
            print(e)
            raise Exception("fail to load model on download_model_from_s3")
    except Exception as e:
        print(e)
        raise Exception("fail to function on download_model_from_s3")

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
    try:
        # model file name
        model_file_name = (
            f"spp_{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl"
        )
        model_local_path = f"{tmp_dir}/{model_file_name}"

        # model to .pkl
        try:
            dump(model, model_local_path)
        except Exception as e:
            print(e)
            raise Exception("fail to dump model on upload_model_to_s3")

        s3 = boto3.client(
            "s3",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        try:
            s3.upload_file(
                model_local_path, aws_s3_bucket_name, f"models/{model_file_name}"
            )
        except Exception as e:
            print(e)
            raise Exception("fail to upload model to s3 on upload_model_to_s3")
    except Exception as e:
        print(e)
        raise Exception("fail to function on upload_model_to_s3")

    return None


if __name__ == "__main__":
    """
    初期化されたモデルを作成
    s3にアップロード
    s3からダウンロード
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()
    tmp_dir: str = "../../tmp/"
    stock_name: str = os.getenv("STOCK_NAME")
    predict_horizon: int = 1
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name: str = os.getenv("AWS_S3_BUCKET_NAME")

    model = PassiveAggressiveRegressor()

    upload_model_to_s3(
        tmp_dir,
        stock_name,
        predict_horizon,
        model,
        aws_region_name,
        aws_access_key_id,
        aws_secret_access_key,
        aws_s3_bucket_name,
    )

    model = download_model_from_s3(
        tmp_dir,
        stock_name,
        predict_horizon,
        aws_region_name,
        aws_access_key_id,
        aws_secret_access_key,
        aws_s3_bucket_name,
    )

    print("finish")
