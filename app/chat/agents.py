from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Callable, Coroutine, Any, Optional, List

import tiktoken
import semantic_kernel as sk
from semantic_kernel.kernel import KernelFunctionBase


ASYNC_CALLABLE = Coroutine[Any, Callable[..., str], str]


class Agent(ABC):

    def __init__(
        self,
        chat_id: Optional[uuid.UUID] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Initializes the Agent class.

        Returns:
            None
        """
        self.kernel = sk.Kernel(*args, **kwargs)
        self.tool_mapping: Dict[str, ASYNC_CALLABLE]
        self.context = self.kernel.create_new_context()
        self._id: uuid.UUID = chat_id or uuid.uuid4()
        self.response: Dict[str, Any] = {'chat_id': str(self._id)}

    async def __call__(self, chat_name: str, prompt: str, *args, **kwargs) -> Dict:
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
        self._config_service(chat_name, *args)
        semantic_function: KernelFunctionBase = await self.prompt(prompt, **kwargs)
        self.context['input'] = prompt
        chat_answer = semantic_function.invoke(context=self.context)
        self.response['output_tokens'] = len(self._encode(chat_answer.variables.variables.get('input', 0)))
        self.response.update(chat_answer.model_dump())
        return self.response

    @abstractmethod
    def _config_service(self, chat_name: str, *args, **kwargs) -> None:
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

    @abstractmethod
    async def _chat_history(self) -> str:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        Args:
            input (str): Name of the deployment to work with
        
        Returns:
            None
        """

    @abstractmethod
    async def _prepare_prompt(self, tool_mapping: Dict[str, Callable[..., str]]) -> str:
        """
        Adds a AI service to the kernel.
        It could be a text completion service, text embedding service or a chat service.

        Args:
            deployment (str): Name of the deployment to work with
        
        Returns:
            None

        Example:
            >>> self.kernel.add_chat_service(
            >>>    schema.deployment_name,
            >>>    AzureChatCompletion(schema.api_key, schema.endpoint)
            >>> )
        """

    @abstractmethod
    async def prompt(self, prompt: str) -> KernelFunctionBase:
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
