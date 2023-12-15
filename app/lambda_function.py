import os
import json
import traceback
from dotenv import load_dotenv
from modules.init_preprocessed_csv import init_preprocessed_csv
from modules.init_model import init_model
from modules.update_model_and_predict import update_model_and_predict


def handler(event, context):
    # read env
    load_dotenv()
    data_dir: str = os.environ["DATA_DIR"]
    models_dir: str = os.environ["MODEL_DIR"]
    target_stock: str = os.environ["TARGET_STOCK"]
    stock_name: str = os.environ["STOCK_NAME"]
    period: str = os.environ["PERIOD"]
    interval: str = os.environ["INTERVAL"]
    predict_horizon: int = 1

    if "handler" in event:
        if event["handler"] == "init":
            try:
                # init preprocess csv
                init_preprocessed_csv(
                    data_dir,
                    stock_name,
                    target_stock,
                    period,
                    interval,
                    predict_horizon,
                )

                # init model
                init_model(data_dir, models_dir, stock_name, predict_horizon)
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(e)
                return {
                    "statusCode": 500,
                    "body": json.dumps(
                        {
                            "message": str(e),
                            "traceback": traceback_str,
                        }
                    ),
                }
        elif event["handler"] == "update":
            # update model and predict
            try:
                update_model_and_predict(
                    data_dir,
                    models_dir,
                    stock_name,
                    target_stock,
                    interval,
                    predict_horizon,
                )
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(e)
                return {
                    "statusCode": 500,
                    "body": json.dumps(
                        {
                            "message": str(e),
                            "traceback": traceback_str,
                        }
                    ),
                }
    else:
        print("invalid handler")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "hello world",
            }
        ),
    }


if __name__ == "__main__":
    handler({"handler": "init"}, None)
    handler({"handler": "update"}, None)
