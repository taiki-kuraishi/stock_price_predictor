import pandas as pd


def shift_dataFrame(df: pd.DataFrame, shift_rows: int) -> pd.DataFrame:
    """
    shift dataFrame
    shift_rows時間後の終値を予測するために、shift_rows行分だけデータをずらす
    """
    df_shifted = df.copy()
    df_shifted["Close"] = df_shifted["Close"].shift(-shift_rows)
    df_shifted.dropna(inplace=True, subset=["Close"])

    return df_shifted
