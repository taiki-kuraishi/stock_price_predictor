"""
This module defines data classes for Lambda configuration,
Stock configuration,
and AWS authentication.
"""
from dataclasses import dataclass
from datetime import tzinfo

from mypy_boto3_dynamodb.service_resource import Table


@dataclass
class LambdaConfiguration:
    """Lambda configuration class."""

    timezone: tzinfo
    tmp_dir: str


@dataclass
class StockConfiguration:
    """Stock configuration class."""

    target_stock: str
    stock_name: str
    period: str
    interval: str


@dataclass
class AWSAuth:
    """AWS authentication class."""

    region_name: str
    access_key_id: str
    secret_access_key: str


@dataclass
class DynamoDBTables:
    """DynamoDB tables class."""

    stock_table: Table
    prediction_table: Table
    limit_table: Table


@dataclass
class ModelsConfiguration:
    """Models configuration class."""

    models_number: int
    dataframe_columns_order: list[str]
    features_columns: list[str]
    target_column: str
