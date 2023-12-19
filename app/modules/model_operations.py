# import joblib
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from .preprocessing_for_prediction import split_target_and_features, shift_dataFrame


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

    # # output model
    # joblib.dump(
    #     model,
    #     f"{tmp_dir}/{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl",
    # )

    print("init_model finish")
    return model


def make_predictions(
    model: PassiveAggressiveRegressor, df: pd.DataFrame, predict_horizon: int
):
    """
    predict
    modelによる予測を行う
    """
    # set target and explanatory variables
    x, _ = split_target_and_features(df)

    # predict
    y_predict = model.predict(x)
    print("predict: " + str(y_predict))

    return y_predict


def init_dynamodb():
    print("init_dynamodb finish")
