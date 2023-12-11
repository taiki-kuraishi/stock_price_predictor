import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
target_stock = os.environ["TARGET_STOCK"]
stock_name = os.environ["STOCK_NAME"]
period = os.environ["PERIOD"]
interval = os.environ["INTERVAL"]

try:
    df = yf.download(target_stock, period=period, interval=interval)
    print("success download data from yfinance")
except Exception as e:
    print("on error yfinance")

# format datetime
try:
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

    print("success format data")
except Exception as e:
    print("on error format data")
    print(e)
    exit()

# save as csv
try:
    df.to_csv("../data/" + stock_name + "_1h_preprocessed.csv")
    print("success save as csv")
except Exception as e:
    print("on error save as csv")
    print(e)
    exit()
