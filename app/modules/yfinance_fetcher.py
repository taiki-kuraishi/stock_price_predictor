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
    try:
        df = yf.download(tickers=target_stock, period=period, interval=interval)
        if df.empty:
            print("fail to get data from yahoo finance")
            return
        else:
            print("success to get data from yahoo finance")
            print("rows : " + str(len(df)))
    except Exception as e:
        print(e)
        print("fail to get data from yahoo finance")
        return

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
    try:
        df = yf.download(
            tickers=target_stock, start=start_date, end=end_date, interval=interval
        )
        if df.empty:
            print("fail to get data from yahoo finance")
            return
        else:
            print("success to get data from yahoo finance")
            print("rows : " + str(len(df)))
    except Exception as e:
        print(e)
        print("fail to get data from yahoo finance")
        return

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

    return df


if __name__ == "__main__":
    import os
    from datetime import datetime, timedelta
    from dotenv import load_dotenv

    load_dotenv(override=True)
    target_stock: str = os.environ["TARGET_STOCK"]
    period: str = os.environ["PERIOD"]
    interval: str = os.environ["INTERVAL"]
    df_col_order: list = os.environ["DTAFRAME_COLUMNS_ORDER"].split(",")

    # get_all_from_yfinance
    df = get_all_from_yfinance(target_stock, period, interval, df_col_order)
    print(df)

    # get_data_for_period_from_yfinance
    today = datetime.today().strftime("%Y-%m-%d")
    today = "2023-12-10"
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    print("today: " + today, "\ntomorrow: " + tomorrow)

    df = get_data_for_period_from_yfinance(
        target_stock, today, tomorrow, interval, df_col_order
    )
    print(df)
