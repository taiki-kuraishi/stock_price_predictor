import pandas as pd


def split_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    split datetime to year, month, day, hour, dayofweek
    """

    df.loc[:, "Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df.loc[:, "year"] = df["Datetime"].dt.year
    df.loc[:, "month"] = df["Datetime"].dt.month
    df.loc[:, "day"] = df["Datetime"].dt.day
    df.loc[:, "hour"] = df["Datetime"].dt.hour
    df.loc[:, "dayofweek"] = df["Datetime"].dt.dayofweek

    df = df[
        [
            "Datetime",
            "year",
            "month",
            "day",
            "hour",
            "dayofweek",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
    ]

    return df


def split_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    split target and features
    目的変数:xと説明変数:yに分割する
    """
    drop_columns = ["Datetime", "Close"]
    y = df["Close"]
    x = df.drop(drop_columns, axis=1)
    return x, y
