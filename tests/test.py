import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../app"))
from lambda_function import lambda_handler

if __name__ == "__main__":
    load_dotenv(dotenv_path="../.env", verbose=True, override=True)
    event = {"handler": "latest"}
    context = {}
    lambda_handler(event, context)
