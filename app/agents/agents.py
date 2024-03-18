from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import tiktoken
import semantic_kernel as sk
from semantic_kernel.kernel import KernelFunction
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from app.tools.memory._abstract import CosmosAbstractMemory
from app.agents.utils import AIServiceSelectorCorrected
from app.settings.types import COMPLETION_TYPE


class Agent(ABC):

    def __init__(
        self,
        *args,
        agent_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> None:
        """
        Initializes the Agent class.

        Returns:
            None
        """

        self._id: uuid.UUID = agent_id or uuid.uuid4()
        self.kernel = sk.Kernel(ai_service_selector=AIServiceSelectorCorrected(), *args, **kwargs)
        self.response: Dict[str, Any] = {'agent_id': str(self._id)}

    async def __call__(
        self,
        prompt: str,
        completions: List[COMPLETION_TYPE],
        config_kwargs={},
        prompt_kwargs={}
    ) -> Dict:
        """
        Configures the service and returns the result of the prompt function.

        Args:
            instructions (str): Instructions for the prompt.
            content (str): Content for the prompt.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            SKFunctionBase: The result of the prompt function.
        """

        self._config_service(completions, **config_kwargs)
        semantic_function: KernelFunction = await self.prompt(str(self._id), **prompt_kwargs)
        chat_answer = await self.kernel.invoke(semantic_function, request=prompt)
        self.response.update({
            'completion_tokens': len(self._encode(str(chat_answer))),
            'response': str(chat_answer)
        })
        return self.response

    def _config_service(
        self,
        completions: List[COMPLETION_TYPE]
    ) -> None:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        This function should be implemented to iterate over each item in 'completions'
        and apply 'self.kernel.add_service' to them.

        Args:
            chat_name (str): Name of the chat service to work with.
            completions (List[COMPLETION_TYPE]): A list of completion services to be added.

        Returns:
            None
        """
        for completion in completions:
            self.kernel.add_service(completion)

    @abstractmethod
    async def prompt(self, *args, **kwargs) -> KernelFunction:
        """
        Creates a semantic function with the given instructions.

        Args:
            instructions (str): Instructions for the semantic function.

        Returns:
            SKFunctionBase: The created semantic function.
        """

    def _encode(self, input: str) -> List[int]:
        """
        Encodes the input using the tiktoken encoder.

        Args:
            input (str): The input to encode.

        Returns:
            List[int]: The encoded input.
        """
        encoder = tiktoken.get_encoding("cl100k_base")
        return encoder.encode(input)


class MemoryAgent(Agent):

    def _long_term_memory(self, memory: CosmosAbstractMemory) -> None:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        Args:
            input (str): Name of the deployment to work with
        
        Returns:
            None
        """
        self.kernel.import_plugin_from_object(
            TextMemoryPlugin(memory),
            "LongTermMemory"
        )

    def _short_term_memory(self, memory: SemanticTextMemory) -> None:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        Args:
            input (str): Name of the deployment to work with

        Returns:
            None
        """
        self.kernel.import_plugin_from_object(
            TextMemoryPlugin(memory),
            "ShortTermMemory"
        )

    def _config_service(
        self,
        completions: List[COMPLETION_TYPE],
        long_term_memory: Optional[CosmosAbstractMemory] = None,
        short_term_memory: Optional[SemanticTextMemory] = None
    ) -> None:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        Args:
            deployment (str): Name of the deployment to work with

        Returns:
            None

        Example:
            >>> self.kernel.add_chat_service(
            >>>    chat_name,
            >>>    AzureChatCompletion(schema.api_key, schema.endpoint)
            >>> )
        """
        super()._config_service(completions)
        if long_term_memory:
            self._long_term_memory(long_term_memory)
        if short_term_memory:
            self._short_term_memory(short_term_memory)
