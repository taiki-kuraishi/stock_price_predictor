"""
This module provides functions for preprocessing data,
specifically for shifting data in a DataFrame and
post-processing stock data retrieved from DynamoDB.

Functions:
- shift_dataframe:
    Shifts the values in a specified column of a DataFrame by a specified number of rows.

- post_process_stock_data_from_dynamodb:
    Post-processes stock data retrieved from DynamoDB.
"""
import pandas as pd


def shift_dataframe(df: pd.DataFrame, shift_rows: int, target_col: str) -> pd.DataFrame:
    """
    データフレームの特定の列の値を指定された行数だけシフトします。

    Parameters
    ----------
    df : pd.DataFrame
        シフトする値を含む元のデータフレーム。
    shift_rows : int
        シフトする行数。
    target_col : str
        シフトする対象の列名。

    Returns
    -------
    df_shifted : pd.DataFrame
        シフトされた値を含む新しいデータフレーム。NaNを含む行は削除されます。
    """
    df_shifted = df.copy()
    df_shifted[target_col] = df_shifted[target_col].shift(-shift_rows)
    df_shifted.dropna(inplace=True, subset=[target_col])

    return df_shifted


def post_process_stock_data_from_dynamodb(
    df: pd.DataFrame,
    df_col_order: list[str],
) -> pd.DataFrame:
    """
    DynamoDBから取得した株価データを後処理します。

    DynamoDBから取得したデータは、カラムの順番や行の順番がバラバラなので、整形します。
    データフレームのインデックスをリセットし、列を指定した順序に並べ替え、日付と時間でソートします。

    Parameters
    ----------
    df : pd.DataFrame
        DynamoDBから取得した株価データ。
    df_col_order : list[str]
        データフレームのカラムの順序を指定するリスト。

    Returns
    -------
    df : pd.DataFrame
        後処理され、カラムと行が整理された株価データを含むデータフレーム。
    """
    df = df.reindex(columns=df_col_order)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values(by=["datetime"])
    df = df.drop(["datetime"], axis=1)
    df = df.reset_index(drop=True)

    return df
