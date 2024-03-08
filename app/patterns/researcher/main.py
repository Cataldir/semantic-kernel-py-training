from typing import List

from ._abstract import AbstractResearcher, DataSource
from .researchers import ConcreteResearcher
from .sources import CosmosDataSource, SQLDataSource, DatabricksDataSource


def client_code(data_sources: List[DataSource], researcher: AbstractResearcher) -> None:
    for source in data_sources:
        source.accept(researcher)
    synthesized_info = researcher.synthesize_information()
    print(f"Synthesized Information: {synthesized_info}")


def main():
    components = [CosmosDataSource(), SQLDataSource(), DatabricksDataSource()]

    print("The client code works with all visitors via the base Visitor interface:")
    visitor1 = ConcreteResearcher()
    client_code(components, visitor1)

    print("It allows the same client code to work with different types of visitors:")
    visitor2 = ConcreteResearcher()
    client_code(components, visitor2)


if __name__ == "__main__":
    main()
