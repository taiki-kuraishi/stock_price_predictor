# %%
# import libraries
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from modules.shift_data_for_prediction import shift_dataFrame
from modules.preprocessing_for_prediction import split_target_and_features

# %%
# constant
PREDICTION_HORIZON = 1

# %%
# read dotenv
load_dotenv()
target_stock = os.environ["TARGET_STOCK"]
stock_name = os.environ["STOCK_NAME"]
period = os.environ["PERIOD"]
interval = os.environ["INTERVAL"]

# %%
# read csv
df = pd.read_csv(
    "../data/" + stock_name + "_1h_preprocessed.csv",
    encoding="utf-8",
    index_col=0,
)
df.head()

# %%
# create shifted data for prediction
df = shift_dataFrame(df, PREDICTION_HORIZON)

# %%
# tarainとtestに分割
split_pos = int(len(df) * 0.7)
train = df.iloc[:split_pos]
test = df.iloc[split_pos:]

# testのindexをリセット
test = test.reset_index(drop=True)

# %%
# 目的変数と説明変数に分割
x_train, y_train = split_target_and_features(train, PREDICTION_HORIZON)
x_test, y_test = split_target_and_features(test, PREDICTION_HORIZON)

print(x_train.head())
print(y_train.head())
print(x_test.head())
print(y_test.head())

# %%
# modeling
from sklearn.linear_model import PassiveAggressiveRegressor

model = PassiveAggressiveRegressor()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# %%
# validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE = %s" % round(mse, 3))
print("MAE = %s" % round(mae, 3))
print("R2 = %s" % round(r2, 3))

# %%
# plot


length = 0
y_test = y_test[length:].values
y_pred = y_pred[length:]

plt.figure(figsize=(16, 9))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()
