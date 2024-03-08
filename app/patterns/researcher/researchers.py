from typing import Optional
import uuid

from semantic_kernel.kernel import KernelFunction

from app.patterns.researcher._abstract import AbstractResearcher, DataSource


class PersistedDataResearcher(AbstractResearcher):

    def __init__(self, *args, chat_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        super().__init__(*args, chat_id, **kwargs)
        self.collected_data = []

    def prepare_query(self, prompt: str, **kwargs) -> KernelFunction:
        prompt_template = """
        You are a proficient data engineer in {{$db_type}}.\n
        You will create a query that is suitable for the technology you are specialized in.\n
        \n------------------------------\n
        Explicitly consider the following data dictionary to provide the best answer:\n
        {{$data_dictionary}}
        \n------------------------------\n
        Use the following examples to help constructing your answer:\n
        *****************\n
        PROMPT: \n
        DATABASE TYPE: Databricks\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        PROMPT: \n
        DATABASE TYPE: CosmosDB\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        PROMPT: \n
        DATABASE TYPE: Redis\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        PROMPT: \n
        DATABASE TYPE: Postgres\n
        DATA DICTIONARY:\n
        
        GENERATED QUERY:\n

        *****************\n
        PROMPT: \n
        DATABASE TYPE: Cassandra\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        \n------------------------------\n
        Provide a summary to a research based on the following question:\n
        {{$input}}
        """
        self.context['input'] = prompt
        return self.kernel.create_semantic_function(prompt_template, **kwargs)

    def visit_data_source(self, source: DataSource, query: str, *args, **kwargs) -> None:
        data = source.provide_data(query, *args, **kwargs)
        self.collected_data.append(data)

    def synthesize_information(self, queried_information: str, **kwargs) -> KernelFunction:
        """
        Synthesize information from all collected data.
        """
        prompt_template = """
        You are a proficient data analyst.\n
        You will create a data summary based on a set of data retrieved from different data sources.\n
        \n------------------------------\n
        Explicitly consider the following data dictionary to provide the best answer:\n
        {{$data_dictionary}}
        \n------------------------------\n
        Use the following examples to help constructing your answer:\n
        *****************\n
        DATABASE TYPE: Databricks\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        DATABASE TYPE: CosmosDB\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        DATABASE TYPE: Redis\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        DATABASE TYPE: Postgres\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        DATABASE TYPE: Cassandra\n
        DATA DICTIONARY:\n

        GENERATED QUERY:\n

        *****************\n
        \n------------------------------\n
        Provide a analytical summary the following dataset:\n
        {{$queried_information}}
        """
        self.context['queried_information'] = queried_information
        return self.kernel.create_semantic_function(prompt_template, **kwargs)
