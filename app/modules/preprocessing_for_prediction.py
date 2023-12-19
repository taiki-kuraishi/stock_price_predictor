import pandas as pd


def split_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    split target and features
    目的変数:xと説明変数:yに分割する
    """
    drop_columns = ["id", "datetime", "close"]
    y = df["close"]
    x = df.drop(drop_columns, axis=1)
    return x, y


def shift_dataFrame(df: pd.DataFrame, shift_rows: int) -> pd.DataFrame:
    """
    shift dataFrame
    shift_rows時間後の終値を予測するために、shift_rows行分だけデータをずらす
    """
    df_shifted = df.copy()
    df_shifted["close"] = df_shifted["close"].shift(-shift_rows)
    df_shifted.dropna(inplace=True, subset=["close"])

    return df_shifted
