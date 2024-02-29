from typing import Optional
import uuid

from app.patterns.researcher._abstract import AbstractResearcher, DataSource


class ConcreteResearcher(AbstractResearcher):

    def __init__(self, *args, chat_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        super().__init__(*args, chat_id, **kwargs)
        self.collected_data = []

    def visit_data_source(self, source: DataSource, *args, **kwargs) -> None:
        data = source.provide_data(*args, **kwargs)
        self.collected_data.append(data)
        print(f"Collected data: {data}")

    def synthesize_information(self) -> str:
        """
        Synthesize information from all collected data.
        """
        return " ".join(self.collected_data)
