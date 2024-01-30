from __future__ import annotations

import uuid
import logging
from typing import Dict
from string import Template

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import KernelFunctionBase

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from app.chat.schemas import ChatSchema, SearchEngineSchema
from app.chat.agents import Agent, ASYNC_CALLABLE
from app.settings.mongo import MongoSettings, MongoAccessor


logger: logging.Logger = logging.getLogger(__name__)


class Researcher(Agent):
    def _config_service(self, chat_name: str, *args, **kwargs) -> None:
        """
        Configures and adds a chat service to the kernel.

        Args:
            chat_name (str): Name of the chat service to configure.
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.
        """
        self.kernel.add_chat_service(
            chat_name,
            AzureChatCompletion(**ChatSchema().model_dump())
        )

    async def _chat_history(self, *args, **kwargs) -> str:
        """
        Retrieves the chat history from the database.

        Args:
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.

        Returns:
            str: The chat history as a string.
        """
        db_settings = MongoSettings()
        db = MongoAccessor('chat', db_settings)
        history = await db.read(collection='chat_history', document_id=uuid.uuid4())
        return history

    async def _prepare_prompt(self, tool_mapping: Dict[str, ASYNC_CALLABLE], *args) -> str:
        """
        Prepares a prompt for the researcher by replacing placeholders with values from async callables.

        Args:
            tool_mapping (Dict[str, ASYNC_CALLABLE]): A mapping of placeholders to asynchronous callable functions.
            *args: Additional arguments to pass to the callable functions.
            **kwargs: Arbitrary keyword arguments for the chat service.

        Returns:
            str: The prepared prompt with placeholders replaced.
        """
        prompt_template = """
        You are a research assistant.\n
        You will guide the user through the process of finding information and preparing the proper summarization of topic research.\n
        Your answer should be structured in topics, based on the content of the chat history and the presaved terms of the research.\n
        Your answer should have at least 1000 words.\n

        \n------------------------------\n

        Consider the following chat history in your answers:\n
        {$CHAT_HISTORY}

        \n------------------------------\n

        Consider the following presaved terms of the research:\n
        {$RESEARCH_TOPICS}

        \n------------------------------\n

        Provide a summary to a research based on the following question:\n
        """
        template = Template(prompt_template)
        results = {}
        for key, function in tool_mapping.items():
            try:
                results[key] = await function(*args)
            except Exception as e:
                print('Error: ', e)
                results[key] = ''
        result = template.substitute(results)
        self.response['input_tokens'] = len(self._encode(result))
        self.response['input_prompt'] = result
        return result

    async def prompt(self, prompt: str, **kwargs) -> KernelFunctionBase:
        """
        Creates and returns a semantic function based on the given prompt and tool mappings.

        Args:
            prompt (str): The prompt to use for creating the semantic function.
            **kwargs: Arbitrary keyword arguments for the semantic function.

        Returns:
            KernelFunctionBase: The created semantic function.
        """
        tool_mapping: Dict[str, ASYNC_CALLABLE] = {}
        tool_mapping['CHAT_HISTORY'] = self._chat_history
        tool_mapping['RESEARCH_TOPICS'] = self.augumented_retrieve
        instructions: str = await self._prepare_prompt(tool_mapping, *[prompt, ])
        return self.kernel.create_semantic_function("%s \n {{$input}}" % instructions, **kwargs)

    async def augumented_retrieve(self, prompt: str) -> str:
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
