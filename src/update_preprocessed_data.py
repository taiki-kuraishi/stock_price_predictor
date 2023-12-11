import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
target_stock = os.environ["TARGET_STOCK"]
stock_name = os.environ["STOCK_NAME"]
interval = os.environ["INTERVAL"]

# read csv and get enc date od preprocess csv
preprocessed_df = (
    pd.read_csv("../data/" + stock_name + "_1h_preprocessed.csv")
    .tail(1)["Datetime"]
    .values[0]
)
print(preprocessed_df)
print(type(preprocessed_df))

# preprocessed_df to YYYY-MM-DD
preprocessed_df = preprocessed_df[:10]
print(preprocessed_df)


# get today
today = datetime.today().strftime("%Y-%m-%d")
print(today)

# get data from yahoo finance
data = yf.download(
    target_stock, start=preprocessed_df, end="2023-12-13", interval=interval
)
print(data)
