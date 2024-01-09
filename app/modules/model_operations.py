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
    # create shifted data for prediction
    df = shift_dataFrame(df, predict_horizon)

    # set target and explanatory variables
    x, y = split_target_and_features(df)

    # model instance
    model = PassiveAggressiveRegressor()

    # train model
    model.fit(x, y)

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
    # create shifted data for prediction
    df = shift_dataFrame(df, predict_horizon)

    # set target and explanatory variables
    x, y = split_target_and_features(df)

    # incremental learning
    model.partial_fit(x, y)

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
