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
    data_dir: str = os.getenv("DATA_DIR")
    models_dir: str = os.getenv("MODEL_DIR")
    target_stock: str = os.getenv("TARGET_STOCK")
    stock_name: str = os.getenv("STOCK_NAME")
    period: str = os.getenv("PERIOD")
    interval: str = os.getenv("INTERVAL")
    predict_horizon: int = 1

    # check env
    if not all([data_dir, models_dir, target_stock, stock_name, period, interval]):
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "fail to read env. not all env are set",
                }
            ),
        }

    # check handler
    if "handler" not in event:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "fail to read handler",
                }
            ),
        }

    # init preprocess csv and model
    if event["handler"] == "init":
        # init preprocess csv
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
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to init preprocess csv",
                        "errorMessage": str(e),
                        "traceback": traceback_str,
                    }
                ),
            }

        # init model
        try:
            # init model
            init_model(data_dir, models_dir, stock_name, predict_horizon)
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "fail to init model",
                        "errorMessage": str(e),
                        "traceback": traceback_str,
                    }
                ),
            }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to init preprocess csv and model",
                }
            ),
        }

    # update model and predict
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
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "success to update model and predict",
                }
            ),
        }
    else:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "wrong handler",
                }
            ),
        }


if __name__ == "__main__":
    handler({"handler": "init"}, None)
    handler({"handler": "update"}, None)
