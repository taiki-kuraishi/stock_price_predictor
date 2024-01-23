import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
# pylint: disable=wrong-import-position
from api_lambda.app.lambda_function import lambda_handler

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../api_lambda/.env.api", verbose=True, override=True)
    print(lambda_handler({"handler": "latest"}, None))
