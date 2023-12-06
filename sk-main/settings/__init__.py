"""
A package that holds settings for the webservice.
"""

__all__: list[str] = ["SQLSettings", "MongoSettings"]
__author__: str = "Ricardo Cataldi"
__copyright__: str = "Copyright 2022, Value Proposition Analyser team"
__credits__: list[str] = ["Ricardo Cataldi <cataldi.ricardo@gmail.com>"]
__version__: str = "0.1.0"
__maintainer__: str = "Value Proposition Analyser team"
__email__: str = "<cataldi.ricardo@gmail.com>"
__status__: str = "In Development"

from .mongo import MongoSettings
from .postgres import SQLSettings