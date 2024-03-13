"""
This module defines data classes for Lambda configuration,
Stock configuration,
and AWS authentication.
"""
from dataclasses import dataclass
from datetime import tzinfo


@dataclass
class StockConfiguration:
    """Stock configuration class."""

    timezone: tzinfo
    saturday: int
    sunday: int
    time_list: list[int]
    dict_order: list[str]
