import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from modules.shift_data_for_prediction import shift_dataFrame
from modules.preprocessing_for_prediction import split_target_and_features
from sklearn.linear_model import PassiveAggressiveRegressor

PREDICTION_HORIZON = 1

load_dotenv()
stock_name = os.environ["STOCK_NAME"]

# read preprocessed csv
try:
    df = pd.read_csv(
        "../data/" + stock_name + "_1h_preprocessed.csv",
        encoding="utf-8",
        index_col=0,
    )
    print("success read csv")
except Exception as e:
    print("on error read csv")
    print(e)
    exit()

# create shifted data for prediction
df = shift_dataFrame(df, PREDICTION_HORIZON)

# set target and explanatory variables
x, y = split_target_and_features(df)

# model construction
model = PassiveAggressiveRegressor()
model.fit(x, y)

# output model
joblib.dump(
    model,
    "../models/"
    + stock_name
    + "_"
    + str(PREDICTION_HORIZON)
    + "h_PassiveAggressiveRegressor.pkl",
)
print("success output model")

print("finish")
