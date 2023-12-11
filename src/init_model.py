import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import PassiveAggressiveRegressor

load_dotenv()
stock_name = os.environ["STOCK_NAME"]

# read preprocessed csv
try:
    df = pd.read_csv("../data/" + stock_name + "_1h_preprocessed.csv", encoding="utf-8")
    print("success read csv")
except Exception as e:
    print("on error read csv")
    print(e)
    exit()

# set target and explanatory variables
try:
    drop_columns = ["Datetime", "Close"]
    x = df.drop(drop_columns, axis=1)
    y = df["Close"]
    print("success set target and explanatory variables")
except Exception as e:
    print("on error set target and explanatory variables")
    print(e)
    exit()

# model construction
try:
    model = PassiveAggressiveRegressor()
    model.fit(x, y)
    print("success model construction")
except Exception as e:
    print("on error model construction")
    print(e)
    exit()

# output model
try:
    joblib.dump(model, "../models/" + stock_name + "_1h_PassiveAggressiveRegressor.pkl")
    print("success output model")
except Exception as e:
    print("on error output model")
    print(e)
    exit()

print("finish")
