import pandas as pd
import yfinance as yf


def get_all_from_yfinance(
    target_stock: str,
    period: str,
    interval: str,
    df_col_order: list,
) -> pd.DataFrame:
    """
    get data from yfinance
    yfinanceからとれるだけすべてのデータを取得する
    """
    # get data from yahoo finance
    df = yf.download(tickers=target_stock, period=period, interval=interval)
    if df.empty:
        print("fail to get data from yahoo finance")
        raise Exception(
            "fail to get data from yahoo finance. data is empty (get_all_from_yfinance)"
        )
    else:
        print("success to get data from yahoo finance")
        print("rows : " + str(len(df)))

    # reset index and rename column
    df.reset_index(inplace=True, drop=False)

    # column name to lower
    df.columns = df.columns.str.lower()

    # add id column
    df["id"] = df.index

    # rename column
    df.rename(columns={"date": "datetime"}, inplace=True)
    df.rename(columns={"adj close": "adj_close"}, inplace=True)

    # sort columns
    df = df.reindex(columns=df_col_order)

    # drop timezone
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    return df


def get_data_for_period_from_yfinance(
    target_stock: str,
    start_date: str,
    end_date: str,
    interval: str,
    df_col_order: list,
) -> pd.DataFrame:
    """
    get data from yfinance
    yfinanceからstart_dateからend_dateまでのデータを取得する
    """
    # get data from yahoo finance
    df = yf.download(
        tickers=target_stock, start=start_date, end=end_date, interval=interval
    )
    if df.empty:
        print("fail to get data from yahoo finance")
        raise Exception(
            "fail to get data from yahoo finance. data is empty (get_data_for_period_from_yfinance)"
        )
    else:
        print("success to get data from yahoo finance")
        print("rows : " + str(len(df)))

    # reset index and rename column
    df.reset_index(inplace=True, drop=False)

    # column name to lower
    df.columns = df.columns.str.lower()

    # add id column
    df["id"] = df.index

    # rename column
    df.rename(columns={"date": "datetime"}, inplace=True)
    df.rename(columns={"adj close": "adj_close"}, inplace=True)

    # sort columns
    df = df.reindex(columns=df_col_order)

    # drop timezone
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    return df
