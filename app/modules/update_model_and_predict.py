import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from .shift_data_for_prediction import shift_dataFrame
from .preprocessing_for_prediction import split_datetime
from .preprocessing_for_prediction import split_target_and_features


def update_model_and_predict(
    csv_dir: str,
    model_dir: str,
    stock_name: str,
    target_stock: str,
    interval: int,
    predict_horizon: int,
) -> None:
    """
    update model and predict
    modelを更新して予測する
    """
    # read preprocessed csv
    preprocessed_csv = pd.read_csv(
        f"{csv_dir}/{stock_name}_{str(predict_horizon)}h_preprocessed.csv",
        encoding="utf-8",
        index_col=0,
    )

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
        if update_df.empty:
            print("fail to get data from yahoo finance")
            return
        else:
            print("success to get data from yahoo finance")
    except Exception as e:
        print(e)
        print("fail to get data from yahoo finance")
        return

    # reset index and rename column
    update_df.reset_index(inplace=True)
    update_df.rename(columns={"Date": "Datetime"}, inplace=True)

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
    concat_df.to_csv(
        f"{csv_dir}/{stock_name}_{str(predict_horizon)}h_preprocessed.csv",
        encoding="utf-8",
    )
    print("success concat data")

    # updateされた件数を出力
    update_rows = len(concat_df) - len(preprocessed_csv)
    print("update rows: " + str(update_rows))

    if update_rows == 0:
        print("no update")
    else:
        # 増分学習用のデータを作成
        train_df = shift_dataFrame(concat_df, predict_horizon).tail(update_rows)
        x, y = split_target_and_features(train_df)

        # model load
        model = joblib.load(
            f"{model_dir}/{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl"
        )

        # model update
        model.fit(x, y)

        # predict
        x_predict = concat_df.tail(1)
        x_predict, _ = split_target_and_features(x_predict)
        y_predict = model.predict(x_predict)
        print("predict: " + str(y_predict))

        # model output
        joblib.dump(
            model,
            f"{model_dir}/{stock_name}_{str(predict_horizon)}h_PassiveAggressiveRegressor.pkl",
        )

    print("update_model_and_predict")
