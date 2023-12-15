import pandas as pd


def shift_dataFrame(df: pd.DataFrame, shift_rows: int) -> pd.DataFrame:
    df_shifted = df.copy()
    df_shifted["Close"] = df_shifted["Close"].shift(-shift_rows)
    df_shifted.dropna(inplace=True, subset=["Close"])

    return df_shifted


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    stock_name = os.environ["STOCK_NAME"]

    df = pd.read_csv(
        "../../data/" + stock_name + "_1h_preprocessed.csv",
        encoding="utf-8",
        index_col=0,
    )

    print(df.tail())

    df_shifted = shift_dataFrame(df, 1)

    print(df_shifted.tail())
