import pandas as pd
from datetime import datetime, timedelta
from yfinance_fetcher import get_data_for_period_from_yfinance


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


def get_updated_data(
    target_stock: str,
    interval: int,
    df_col_order: list,
    df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    update model and predict
    dynamodbに保存されているデータの最終更新日時をもとに、新たにデータを取得する
    新たなデータがない場合は、Noneを返す
    """

    # get last_updated_datetime YYYY-MM-DD HH:MM:SS
    last_updated_datetime = df.tail(1)["datetime"].values[0]
    # last_updated_datetime = "2023-12-10 15:30:00-05:00"
    print("last updated datetime: " + last_updated_datetime)

    # last_updated_datetime to YYYY-MM-DD
    last_updated_date = last_updated_datetime[:10]
    print("last updated date: " + last_updated_date)

    # get tomorrow
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    print("tomorrow: " + tomorrow)

    # get data from yahoo finance
    update_df = get_data_for_period_from_yfinance(
        target_stock, last_updated_date, tomorrow, interval, df_col_order
    )

    # update_dfの[datetime]がlast_updated_dateより前の場合は、削除する
    update_df = update_df[update_df["datetime"] > last_updated_datetime]

    if update_df.empty:
        print("no update")
        return None

    # get last index from preprocessed csv
    last_index = df.tail(1)["id"].values[0]
    print("last index: " + str(last_index))

    # add id
    update_df["id"] = update_df.index + last_index + 1

    # updateされた件数を出力
    update_rows = len(update_df)
    print("update rows: " + str(update_rows))

    return update_df