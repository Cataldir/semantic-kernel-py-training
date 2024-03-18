from typing import Any, Dict

from pydantic import BaseModel
from ._abstract import DataSource, AbstractResearcher

from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class CosmosDataSource(DataSource):
    """
    Extracts data from a CosmosDB instance.
    """

    @kernel_function(name="extract_from_cosmos", description="Extracts data from a CosmosDB instance.")
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return {'data': "Data from Source A"}

    def accept(self, researcher: AbstractResearcher, query: str) -> None:
        researcher.visit_data_source(self, query)


class SQLDataSource(DataSource):
    """
    Extracts data from a Azure SQL instance.
    """

    @kernel_function(name="extract_from_sql", description="Extracts data from a Azure SQL instance.")
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return {'data': "Data from Source A"}

    def accept(self, researcher: AbstractResearcher, query: str) -> None:
        researcher.visit_data_source(self, query)


class DatabricksDataSource(DataSource):
    """
    Extracts data from a Azure Databricks instance.
    """

    @kernel_function(name="extract_from_cosmos", description="Extracts data from a CosmosDB instance.")
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return {'data': "Data from Source A"}

    def accept(self, researcher: AbstractResearcher, query: str) -> None:
        researcher.visit_data_source(self, query)


class BlobDataSource(DataSource):
    """
    Extracts data from a BlobStorage instance.
    """

    @kernel_function(name="extract_from_cosmos", description="Extracts data from a CosmosDB instance.")
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        return {'data': "Data from Source A"}

    def accept(self, researcher: AbstractResearcher, query: str) -> None:
        researcher.visit_data_source(self, query)
