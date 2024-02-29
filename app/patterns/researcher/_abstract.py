from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

from app.agents import Agent


class DataSource(ABC):
    """
    The DataSource interface represents a data source from which information
    can be retrieved. Each data source is a component that the Researcher can
    visit.
    """

    @abstractmethod
    def provide_data(self, query: str, params: BaseModel) -> Dict[str, Any]:
        """
        The provide data method returns data from the data source.

        Args:
            query (str): Query to be executed to get the data from the data source.
            params (BaseModel): Parameters to be used to conect to the data source.

        Returns:
            Dict[str, Any]: a set of retrieved information, described as a dictionary.
        """

    @abstractmethod
    def accept(self, researcher: AbstractResearcher) -> None:
        pass


class AbstractResearcher(Agent):
    """
    The Researcher interface extends the agent behaviour to add the multiple retrieval sources.
    """
    @abstractmethod
    def synthesize_information(self) -> str:
        pass

    @abstractmethod
    def visit_data_source(self, source: DataSource) -> None:
        pass
