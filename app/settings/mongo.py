import os
from typing import Optional

from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv
from motor.core import AgnosticClient, AgnosticDatabase
from motor.motor_asyncio import AsyncIOMotorClient

from .base import BaseConnection, SettingsMeta

load_dotenv(find_dotenv())


@dataclass
class MongoSettings(BaseConnection, metaclass=SettingsMeta):
    """
    Database connection settings component

    connects to a PostGres database using asyncpg
    """

    engine: Optional[str] = field(default=os.environ.get("DC_ENGINE", "mongodb"))
    host: Optional[str] = field(default=os.environ.get("DC_HOST", "localhost"))
    port: Optional[str] = field(default=os.environ.get("DC_PORT", "27017"))

    async def connect(self, connection_string: Optional[str] = None) -> AgnosticClient:
        """
        connect to the database that is defined by the settings.
        """
        if not connection_string:
            connection_string = f"{self.engine}://{self.host}:{self.port}"
        connection: AgnosticClient = AsyncIOMotorClient(connection_string)
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
