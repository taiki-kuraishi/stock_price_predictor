import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../app"))
from lambda_function import handler

if __name__ == "__main__":
    load_dotenv(dotenv_path="../.env", override=True, verbose=True)
    # handler({"handler": ""}, None)
    # handler({"handler": "init_stock_table_from_s3"}, None)
    # handler({"handler": "init_stock_table_from_yfinance"}, None)
    # handler({"handler": "delete_pred_table_item"}, None)
    # handler({"handler": "init_model"}, None)
    # handler({"handler": "update_predict"}, None)
    # handler({"handler": "update_stock_table"}, None)
    # handler({"handler": "update_model"}, None)
