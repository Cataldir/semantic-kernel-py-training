from __future__ import annotations

import uuid
import logging
from typing import Type

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import KernelFunctionBase

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from pymongo.errors import ServerSelectionTimeoutError

from app.agents.agents import Agent
from app.schemas.agents import ChatSchema, SearchEngineSchema
from app.settings.mongo import MongoSettings, MongoAccessor
from app.utils.tracker import evaluate_performance


logger: logging.Logger = logging.getLogger(__name__)


class Researcher(Agent):

    def _config_service(
        self,
        chat_name: str,
        completion: Type[AzureChatCompletion] = AzureChatCompletion,
        schema: ChatSchema = ChatSchema()
    ) -> None:
        """
        Configures and adds a chat service to the kernel.

        Args:
            chat_name (str): Name of the chat service to configure.
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.
        """
        self.kernel.add_chat_service(
            chat_name,
            completion(**schema.model_dump())
        )

    async def _chat_history(
        self,
        overlapping_percent: float = 0,
        max_messages: int = 10
    ) -> str:
        """
        Retrieves the chat history from the database.

        Args:
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.

        Returns:
            str: The chat history as a string.
        """
        try:
            db_settings = MongoSettings()
            db = MongoAccessor('chat', db_settings)
            history = await db.read(collection='chat_history', document_id=uuid.uuid4())
        except ServerSelectionTimeoutError as e:
            logger.error(f'Error retrieving chat history: {e}')
            history = ''
        return history

    @evaluate_performance
    async def prompt(self, prompt: str, **kwargs) -> KernelFunctionBase:
        """
        Creates and returns a semantic function based on the given prompt and tool mappings.

        Args:
            prompt (str): The prompt to use for creating the semantic function.
            **kwargs: Arbitrary keyword arguments for the semantic function.

        Returns:
            KernelFunctionBase: The created semantic function.
        """

        prompt_template = """
        You are a research assistant.\n
        You will write a summary of the research, with a brief introduction and a review of the topic.\n
        Your answer should be structured in topics, based on the content of the chat history and the presaved terms of the research.\n
        Your answer should have at least 1000 words.\n
        \n------------------------------\n
        Consider the following chat history in your answers:\n
        {{$CHAT_HISTORY}}
        \n------------------------------\n
        Consider the following presaved researched documents:\n
        {{$RESEARCH_TOPICS}}
        \n------------------------------\n
        Provide a summary to a research based on the following question:\n
        {{$input}}
        """

        chat_history_params = {
            k: kwargs.pop(k)
            for k in list(kwargs.keys())
            if k in ['overlapping_percent', 'max_messages']
        }

        self.context['CHAT_HISTORY'] = await self._chat_history(**chat_history_params)
        self.context['RESEARCH_TOPICS'] = await self.augmented_retrieve(prompt)
        return self.kernel.create_semantic_function(prompt_template, **kwargs)

    async def augmented_retrieve(self, prompt: str) -> str:
        """
        Performs an augmented retrieval of research documents based on the provided prompt.

        Args:
            prompt (str): The prompt to use for the retrieval.

        Returns:
            str: Aggregated content of the top relevant research documents.
        """
        schema = SearchEngineSchema()
        search_engine = SearchClient(
            endpoint=schema.endpoint,
            index_name=schema.index_name,
            credential=AzureKeyCredential(schema.search_key)
        )
        research_docs = ''
        async with search_engine:
            results = await search_engine.search(search_text=prompt, top=10)
            async for result in results:
                if result.get('@search.score', 0) > 20:
                    research_docs += result['content'] + '\n'
        return research_docs
