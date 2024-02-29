"""
Defines a base metaclass for all settings that is thread-safe.

Returns:
    SettingsMeta: A thread-safe metaclass for the settings. 
"""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Protocol, Optional, Callable, Any
from threading import Lock


class SettingsMeta(ABCMeta):
    """
    This is a thread-safe implementation of the settings.
    """

    _instances: dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class BaseConnection(ABC, metaclass=SettingsMeta):
    @abstractmethod
    async def connect(self):
        """
        connects to a database
        """

    @abstractmethod
    def database(self, name: Optional[str]):
        """
        Retrieves a database based on the current connection

        Args:
            name (str): _description_

        Returns:
            AgnosticDatabase: _description_
        """


class DataclassProtocol(Protocol):
    __dataclass_fields__: Dict
    __dataclass_params__: Dict
    __post_init__: Optional[Callable[..., Any]]
