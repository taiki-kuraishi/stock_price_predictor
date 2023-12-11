import os
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from dotenv import load_dotenv
from modules.shift_data_for_prediction import shift_dataFrame
from modules.preprocessing_for_prediction import split_datetime
from modules.preprocessing_for_prediction import split_target_and_features

PREDICTION_HORIZON = 1

# load env
load_dotenv()
target_stock = os.environ["TARGET_STOCK"]
stock_name = os.environ["STOCK_NAME"]
interval = os.environ["INTERVAL"]

# read preprocessed csv
try:
    preprocessed_csv = pd.read_csv(
        "../data/" + stock_name + "_1h_preprocessed.csv",
        encoding="utf-8",
        index_col=0,
    )
    print("success read csv")
except Exception as e:
    print("on error read csv")
    print(e)
    exit()

# get last_updated_date from preprocessed csv
last_updated_date = preprocessed_csv.tail(1)["Datetime"].values[0]

# last_updated_date to YYYY-MM-DD
last_updated_date = last_updated_date[:10]
print("last updated date: " + last_updated_date)

# get tomorrow
tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
print("tomorrow: " + tomorrow)

# get data from yahoo finance
try:
    update_df = yf.download(
        target_stock, start=last_updated_date, end=tomorrow, interval=interval
    )
    print("success download data from yfinance")
except Exception as e:
    print("on error yfinance")
    print(e)
    exit()


# format update_df
data = split_datetime(update_df)


# concat preprocessed_csv and update_df
concat_df = pd.concat(
    [preprocessed_csv, data],
    axis=0,
    join="inner",
    ignore_index=True,
)
concat_df = concat_df.drop_duplicates()
concat_df = concat_df.reset_index(drop=True)
concat_df.to_csv("../data/" + stock_name + "_1h_preprocessed.csv")
print("success concat data")

# updateされた件数を出力
update_rows = len(concat_df) - len(preprocessed_csv)
print("update rows: " + str(update_rows))

if update_rows == 0:
    print("no update")
    exit()

# # updateされたデータを格納
# update_df = concat_df.tail(update_rows)

# 増分学習用のデータを作成
train_df = shift_dataFrame(concat_df, PREDICTION_HORIZON).tail(update_rows)
x, y = split_target_and_features(train_df)

# model load
model = joblib.load(
    "../models/"
    + stock_name
    + "_"
    + str(PREDICTION_HORIZON)
    + "h_PassiveAggressiveRegressor.pkl"
)

# model update
model.fit(x, y)

# predict
x_predict = concat_df.tail(1)
print(x_predict)
x_predict = split_datetime(x_predict)
x_predict, _ = split_target_and_features(x_predict)
y_predict = model.predict(x_predict)

print("predict: " + str(y_predict))


# model output
joblib.dump(
    model,
    "../models/"
    + stock_name
    + "_"
    + str(PREDICTION_HORIZON)
    + "h_PassiveAggressiveRegressor.pkl",
)

print("finish")
