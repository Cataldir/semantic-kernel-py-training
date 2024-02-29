from typing import Any, Dict

from pydantic import BaseModel
from ._abstract import DataSource, AbstractResearcher


class CosmosDataSource(DataSource):
    """
    Extracts data from a CosmosDB instance.
    """
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        pass

    def accept(self, researcher: AbstractResearcher) -> None:
        researcher.visit_data_source(self)


class SQLDataSource(DataSource):
    """
    Extracts data from a Azure SQL instance.
    """
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return "Data from Source B"

    def accept(self, researcher: AbstractResearcher) -> None:
        researcher.visit_data_source(self)


class DatabricksDataSource(DataSource):
    """
    Extracts data from a Azure Databricks instance.
    """
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return "Data from Source B"

    def accept(self, researcher: AbstractResearcher) -> None:
        researcher.visit_data_source(self)


class BlobDataSource(DataSource):
    """
    Extracts data from a BlobStorage instance.
    """
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return "Data from Source B"

    def accept(self, researcher: AbstractResearcher) -> None:
        researcher.visit_data_source(self)
