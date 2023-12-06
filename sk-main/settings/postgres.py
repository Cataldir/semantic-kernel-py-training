import os
from dataclasses import dataclass, field

import asyncpg
from dotenv import find_dotenv, load_dotenv

from .base import BaseConnection, SettingsMeta

load_dotenv(find_dotenv())


@dataclass
class SQLSettings(BaseConnection, metaclass=SettingsMeta):
    """
    Database connection settings component

    connects to a PostGres database using asyncpg
    """

    engine: str = field(default=os.environ.get("DB_ENGINE", "postgres"))
    host: str = field(default=os.environ.get("DB_HOST", "localhost"))
    database: str = field(default=os.environ.get("DB_NAME", "postgres"))
    user: str = field(default=os.environ.get("DB_USER", "postgres"))
    password: str = field(
        default=os.environ.get("DB_PASSWORD", "postgrespw")
    )
    port: str = field(default=os.environ.get("DB_PORT", "5432"))
    base_schema: str = field(
        default=os.environ.get("BASE_SCHEMA", "base_schema")
    )
    user_schema: str = field(
        default=os.environ.get("USER_SCHEMA", "user_schema")
    )
    value_proposition_schema: str = field(
        default=os.environ.get("VPC_SCHEMA", "vpc_schema")
    )

    async def connect(self) -> asyncpg.Connection:
        """
        connect to the database that is defined by the settings.
        """
        connection: asyncpg.Connection = await asyncpg.connect(
            user=self.user,
            password=self.password,
            database=self.database,
            host=self.host,
            port=int(self.port),
        )
        return connection

    async def pool(self) -> asyncpg.Pool:
        """
        pool retrieves a connection pool from asyncpg

        Returns:
            asyncpg.Pool: a uninstantialized pool
        """
        pool: asyncpg.Pool = await asyncpg.create_pool(self.database_url())
        return pool

    async def database_url(self) -> str:
        """
        database_url generates the url for connecting to postgres database

        Returns:
            str: the url for database connection
        """
        database_url: str = (
            f"{self.engine}"
            f"://{self.user}"
            f":{self.password}"
            f"@{self.host}"
            f":{self.port}"
            f"/{self.database}"
        )
        return database_url
