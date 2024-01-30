"""
This module is used for testing the lambda function in the ml_lambda app.
"""
import os
import sys

from aws_lambda_context import (
    LambdaClientContext,
    LambdaClientContextMobileClient,
    LambdaCognitoIdentity,
    LambdaContext,
)
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml_lambda/app/"))
# pylint: disable=wrong-import-position
# pylint: disable=import-error
from ml_lambda.app.lambda_function import handler

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../ml_lambda/.env.ml", override=True, verbose=True)

    lambda_cognito_identity = LambdaCognitoIdentity()
    lambda_cognito_identity.cognito_identity_id = "cognito_identity_id"
    lambda_cognito_identity.cognito_identity_pool_id = "cognito_identity_pool_id"

    lambda_client_context_mobile_client = LambdaClientContextMobileClient()
    lambda_client_context_mobile_client.installation_id = "installation_id"
    lambda_client_context_mobile_client.app_title = "app_title"
    lambda_client_context_mobile_client.app_version_name = "app_version_name"
    lambda_client_context_mobile_client.app_version_code = "app_version_code"
    lambda_client_context_mobile_client.app_package_name = "app_package_name"

    lambda_client_context = LambdaClientContext()
    lambda_client_context.client = lambda_client_context_mobile_client
    lambda_client_context.custom = {"custom": True}
    lambda_client_context.env = {"env": "test"}

    lambda_context = LambdaContext()
    lambda_context.function_name = "function_name"
    lambda_context.function_version = "function_version"
    lambda_context.invoked_function_arn = "invoked_function_arn"
    lambda_context.memory_limit_in_mb = 300
    lambda_context.aws_request_id = "aws_request_id"
    lambda_context.log_group_name = "log_group_name"
    lambda_context.log_stream_name = "log_stream_name"
    lambda_context.identity = lambda_cognito_identity
    lambda_context.client_context = lambda_client_context

    handler({"handler": ""}, lambda_context)
    print(handler({"handler": "update_predict"}, lambda_context))
    print(handler({"handler": "update_stock_table"}, lambda_context))
    print(handler({"handler": "update_and_predict_tables"}, lambda_context))
    print(handler({"handler": "update_model"}, lambda_context))
