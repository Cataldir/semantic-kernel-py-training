from __future__ import annotations

import uuid
from abc import abstractmethod
from typing import Any, Dict

from motor.core import AgnosticDatabase
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemoryBase

from app.settings import MongoSettings


class CosmosAbstractMemory(SemanticTextMemoryBase):

    settings: MongoSettings
    database: AgnosticDatabase

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Async close connection, invoked by MemoryStoreBase.__aexit__()"""

    @abstractmethod
    async def create(self, dataclass_instance: Dict[str, Any]):
        """
        _summary_

        Args:
            dataclass_instance (DataclassProtocol): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    async def read(self, collection: str, document_id: uuid.UUID):
        """
        _summary_

        Args:
            collection (str): _description_
            document_id (uuid.UUID): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    async def update(self, document_id: uuid.UUID, dataclass_instance: Dict[str, Any]):
        """
        _summary_

        Args:
            document_id (uuid.UUID): _description_
            dataclass_instance (DataclassProtocol): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    async def delete(self, collection: str, document_id: uuid.UUID):
        """
        _summary_

        Args:
            collection (str): _description_
            document_id (uuid.UUID): _description_

        Returns:
            _type_: _description_
        """
