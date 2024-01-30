"""
A package that holds settings for the webservice.
"""

__all__: list[str] = ["SQLSettings", "MongoSettings"]
__author__: str = "Ricardo Cataldi"
__version__: str = "0.1.0"
__status__: str = "In Development"

from .mongo import MongoSettings
from .postgres import SQLSettings