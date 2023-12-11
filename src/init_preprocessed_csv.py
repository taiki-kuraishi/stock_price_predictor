import os
import yfinance as yf
from dotenv import load_dotenv
from modules.preprocessing_for_prediction import split_datetime

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
df = split_datetime(df)

# save as csv
df.to_csv("../data/" + stock_name + "_1h_preprocessed.csv", encoding="utf-8")

print("finish")
