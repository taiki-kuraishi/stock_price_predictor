import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from .dataframe_operations import shift_dataFrame, split_target_and_features


# モデルの学習
def init_and_retrain_model(
    df: pd.DataFrame, predict_horizon: int
) -> PassiveAggressiveRegressor:
    """
    init model
    modelの初期化と学習を行う
    """
    try:
        # create shifted data for prediction
        df = shift_dataFrame(df, predict_horizon)

        # set target and explanatory variables
        x, y = split_target_and_features(df)

        # model instance
        model = PassiveAggressiveRegressor()

        # train model
        model.fit(x, y)
    except Exception as e:
        print(e)
        raise Exception("fail to function on init_and_retrain_model")

    return model


# 増分学習
def incremental_learning(
    model: PassiveAggressiveRegressor,
    df: pd.DataFrame,
    predict_horizon: int,
) -> PassiveAggressiveRegressor:
    """
    incremental learning
    増分学習を行う
    """
    try:
        # create shifted data for prediction
        df = shift_dataFrame(df, predict_horizon)

        # set target and explanatory variables
        x, y = split_target_and_features(df)

        # incremental learning
        model.partial_fit(x, y)
    except Exception as e:
        print(e)
        raise Exception("fail to function on incremental_learning")

    return model


# 予測
def make_predictions(
    model: PassiveAggressiveRegressor, df: pd.DataFrame
) -> list[float]:
    """
    predict
    modelによる予測を行う
    """
    try:
        # set target and explanatory variables
        x, _ = split_target_and_features(df)

        # predict
        y_predict = model.predict(x)
    except Exception as e:
        print(e)
        raise Exception("fail to function on make_predictions")

    return y_predict


if __name__ == "__main__":
    """
    dynamodbから学習用のデータをダウンロード
    3つのモデルを使用
    モデルの初期化と学習
    モデルの予測
    """
    import os
    from dotenv import load_dotenv
    from dynamodb_fetcher import get_data_from_dynamodb
    from dataframe_operations import post_process_stock_data_from_dynamodb

    load_dotenv(dotenv_path="../../.env", override=True)
    stock_name: str = os.getenv("STOCK_NAME")
    df_col_order: list = os.getenv("DTAFRAME_COLUMNS_ORDER").split(",")
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    dynamodb_stock_table_name = "spp_" + stock_name

    # modelの数
    model_num = 3

    # get stock data
    df = get_data_from_dynamodb(
        aws_region_name,
        aws_access_key_id,
        aws_secret_access_key,
        dynamodb_stock_table_name,
    )
    df = post_process_stock_data_from_dynamodb(df, df_col_order)

    # 予測したデータを格納するdfの作成
    df_predict = pd.DataFrame()
    df_predict["datetime"] = df[["datetime"]]

    for i in range(1, model_num + 1):
        # train model
        model = init_and_retrain_model(df, i)

        # predict
        y_predict = make_predictions(model, df)

        # concat predict to df_predict
        df_predict[f"{i}_pred"] = y_predict

    print(df_predict)

    print("model operation complete")
