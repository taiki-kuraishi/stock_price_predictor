import joblib
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from .shift_data_for_prediction import shift_dataFrame
from .preprocessing_for_prediction import split_target_and_features


def init_model(csv_dir: str, model_dir, stock_name: str, predict_horizon: int) -> None:
    """
    init model
    modelの初期化
    """
    # read preprocessed csv
    df = pd.read_csv(
        f"{csv_dir}/{stock_name}_{str(predict_horizon)}h_preprocessed.csv",
        encoding="utf-8",
        index_col=0,
    )
    print("success to read csv")

    # create shifted data for prediction
    df = shift_dataFrame(df, predict_horizon)
    print("success to create shifted data")

    # set target and explanatory variables
    x, y = split_target_and_features(df)
    print("success to split target and features")

    # model construction
    model = PassiveAggressiveRegressor()
    model.fit(x, y)
    print("success to fit model")

    # output model
    joblib.dump(
        model,
        f"{model_dir}/{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl",
    )

    print("init_model finish")
