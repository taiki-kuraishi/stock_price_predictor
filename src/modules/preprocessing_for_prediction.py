import pandas as pd


def split_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    split datetime to year, month, day, hour, dayofweek
    """
    df = df.reset_index()

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["day"] = df["Datetime"].dt.day
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek

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

    df = df.sort_values("Datetime")

    df = df.reset_index(drop=True)

    return df


def split_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    split target and features
    """
    drop_columns = ["Datetime", "Close"]
    y = df["Close"]
    x = df.drop(drop_columns, axis=1)
    return x, y
