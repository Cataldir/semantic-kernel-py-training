import os
import uuid
from typing import Any

from dataclasses import dataclass, field, asdict

from dotenv import find_dotenv, load_dotenv
from pymongo.results import DeleteResult, UpdateResult, InsertOneResult
from motor.core import AgnosticClient, AgnosticDatabase
from motor.motor_asyncio import AsyncIOMotorClient

from .base import BaseConnection, SettingsMeta, DataclassProtocol

load_dotenv(find_dotenv())


@dataclass
class MongoSettings(BaseConnection, metaclass=SettingsMeta):
    """
    Database connection settings component

    connects to a PostGres database using asyncpg
    """

    engine: str = field(default=os.environ.get("DB_ENGINE", "mongodb"))
    host: str = field(default=os.environ.get("DB_HOST", "localhost"))
    port: str = field(default=os.environ.get("DB_PORT", "27017"))

    def connect(self) -> AgnosticClient:
        """
        connect to the database that is defined by the settings.
        """
        connection: AgnosticClient = AsyncIOMotorClient(
            f"{self.engine}://{self.host}:{self.port}"
        )
        return connection

    def database(self, name: str) -> AgnosticDatabase:
        """
        Retrieves a database based on the current connection

        Args:
            name (str): _description_

        Returns:
            AgnosticDatabase: _description_
        """
        client: AgnosticClient = self.connect()
        return client[name]


class MongoAccessor:
    def __init__(self, database: str, *args) -> None:
        settings: MongoSettings = MongoSettings(*args)
        self.database: AgnosticDatabase = settings.database(database)

    async def create(self, dataclass_instance: DataclassProtocol) -> Any:
        document: dict[str, Any] = asdict(dataclass_instance)
        for key, value in document.items():
            if isinstance(value, uuid.UUID):
                document[key] = str(value)
        result: InsertOneResult = await self.database[
            dataclass_instance.__class__.__name__
        ].insert_one(document)
        return result.inserted_id

    async def read(self, collection: str, document_id: uuid.UUID):
        result = await self.database[collection].find_one({"_id": document_id})
        return result

    async def update(self, document_id: uuid.UUID, dataclass_instance: DataclassProtocol):
        result: UpdateResult = await self.database[
            dataclass_instance.__class__.__name__
        ].update_one({"_id": document_id}, {"$set": asdict(dataclass_instance)})
        return result.modified_count

    async def delete(self, collection: str, document_id: uuid.UUID):
        result: DeleteResult = await self.database[collection].delete_one({"_id": document_id})
        return result.deleted_count
