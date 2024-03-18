from __future__ import annotations

from typing import Callable, Coroutine, Any, TypeVar, Union

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding


AzureChatCompletionT = TypeVar("AzureChatCompletionT", bound=AzureChatCompletion)
AzureTextEmbeddingT = TypeVar("AzureTextEmbeddingT", bound=AzureTextEmbedding)

ASYNC_CALLABLE = Coroutine[Any, Callable[..., str], str]
COMPLETION_TYPE = Union[AzureChatCompletionT, AzureTextEmbeddingT]
