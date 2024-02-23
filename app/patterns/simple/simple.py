from __future__ import annotations

import uuid
import logging
from typing import Type, Optional

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import KernelFunction

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from app.agents.agents import Agent
from app.schemas.agents import ChatSchema, SearchEngineSchema
from app.tools.memories import CosmosMongoMemory
from app.utils.tracker import evaluate_performance


logger: logging.Logger = logging.getLogger(__name__)


class SimpleRAG(Agent):

    def _config_service(
        self,
        chat_name: str,
        completion: Type[AzureChatCompletion] = AzureChatCompletion,
        schema: ChatSchema = ChatSchema(),
        memory: Optional[CosmosMongoMemory] = None
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
        if memory:
            self._chat_history(memory)

    @evaluate_performance
    async def prompt(self, prompt: str, **kwargs) -> KernelFunction:
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
        Consider the following chat history:\n
        {{$chat_history}}
        \n------------------------------\n
        Consider the following presaved researched documents:\n
        {{$RESEARCH_TOPICS}}
        \n------------------------------\n
        Provide a summary to a research based on the following question:\n
        {{$input}}
        """
    
        self.context['chat_history'] = await self.kernel.memory.search('ragMemory', prompt, 10)
        self.context['RESEARCH_TOPICS'] = await self.augmented_retrieve(prompt)
        self.context['input'] = prompt
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
