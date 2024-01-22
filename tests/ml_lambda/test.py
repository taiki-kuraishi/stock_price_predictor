import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml_lambda/app/"))
# pylint: disable=wrong-import-position
from ml_lambda.app.lambda_function import handler

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../ml_lambda/.env", override=True, verbose=True)
    handler({"handler": ""}, None)
    print(handler({"handler": "init_limit_table"}, None))
    print(handler({"handler": "init_stock_table"}, None))
    print(handler({"handler": "delete_pred_table_item"}, None))
    print(handler({"handler": "init_model"}, None))
    print(handler({"handler": "update_predict"}, None))
    print(handler({"handler": "update_stock_table"}, None))
    print(handler({"handler": "update_and_predict_tables"}, None))
    print(handler({"handler": "update_model"}, None))
