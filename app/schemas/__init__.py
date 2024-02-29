"""
A package that holds response schemas and models.
"""

__all__ = ["BodyMessage", "RESPONSES", "ChatEndpoint", "ChatEndpointWithMemory"]
__author__ = "Ricardo Cataldi"
__version__ = "0.1.0"
__status__ = "In Development"

from .responses import RESPONSES, BodyMessage
from .requests import ChatEndpoint, ChatEndpointWithMemory