import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from preprocessing_for_prediction import shift_dataFrame, split_target_and_features


# モデルの学習
def train_model(df: pd.DataFrame, predict_horizon: int) -> PassiveAggressiveRegressor:
    """
    init model
    modelの初期化と学習を行う
    """
    # create shifted data for prediction
    df = shift_dataFrame(df, predict_horizon)

    # set target and explanatory variables
    x, y = split_target_and_features(df)

    # model construction
    model = PassiveAggressiveRegressor()
    model.fit(x, y)

    print("train model complete")
    return model


# 予測
def make_predictions(
    model: PassiveAggressiveRegressor, df: pd.DataFrame
) -> list[float]:
    """
    predict
    modelによる予測を行う
    """
    # set target and explanatory variables
    x, _ = split_target_and_features(df)

    # predict
    y_predict = model.predict(x)

    return y_predict


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from dynamodb_fetcher import get_data_from_dynamodb, upload_data_to_dynamodb

    load_dotenv(dotenv_path="../../.env", override=True)
    df_col_order: list = os.getenv("DTAFRAME_COLUMNS_ORDER").split(",")
    predict_horizon: int = 1
    aws_region_name: str = os.getenv("AWS_REGION_NAME")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_dynamodb_train_table_name: str = os.getenv("AWS_DYNAMODB_TRAIN_TABLE_NAME")
    aws_dynamo_prediction_table_name: str = os.getenv(
        "AWS_DYNAMODB_PREDICTION_TABLE_NAME"
    )

    # get train data
    df = get_data_from_dynamodb(
        df_col_order,
        aws_region_name,
        aws_access_key_id,
        aws_secret_access_key,
        aws_dynamodb_train_table_name,
    )

    # train model
    model = train_model(df, predict_horizon)

    # predict
    y_predict = make_predictions(model, df)

    # 新しいdfの作成
    df_predict = pd.DataFrame()
    df_predict["datetime"] = df[["datetime"]]

    # concat predict to df_predict
    df_predict[f"{predict_horizon}_pred"] = y_predict

    print(df_predict)

    # # upload to dynamodb
    # upload_data_to_dynamodb(
    #     aws_region_name,
    #     aws_access_key_id,
    #     aws_secret_access_key,
    #     aws_dynamo_prediction_table_name,
    #     df_predict,
    # )

    print("model operation complete")
