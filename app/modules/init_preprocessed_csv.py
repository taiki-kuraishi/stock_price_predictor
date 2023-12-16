import yfinance as yf
from .preprocessing_for_prediction import split_datetime


def init_preprocessed_csv(
    csv_dir: str,
    stock_name: str,
    target_stock: str,
    period: str,
    interval: str,
    predict_horizon: int,
) -> None:
    """
    init preprocess csv
    学習用データのcsvを初期化する
    """
    # get data from yahoo finance
    try:
        df = yf.download(tickers=target_stock, period=period, interval=interval)
        print(df)
        if df.empty:
            print("fail to get data from yahoo finance")
            return
        else:
            print("success to get data from yahoo finance")
    except Exception as e:
        print(e)
        print("fail to get data from yahoo finance")
        return

    # reset index and rename column
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Datetime"}, inplace=True)

    # format datetime
    splitted_datetime_df = split_datetime(df)

    # save as csv
    splitted_datetime_df.to_csv(
        f"{csv_dir}/{stock_name}_{str(predict_horizon)}h_preprocessed.csv",
        encoding="utf-8",
    )

    print("init_preprocessed_csv finish")
