from typing import List

from numpy import ndarray

from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)

import tiktoken


class GPTEmbeddingGenerator(EmbeddingGeneratorBase):
    """
    Azure Embedding Generator
    """
    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def generate_embeddings(self, texts: List[str]) -> ndarray:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The generated embedding.
        """
        return ndarray([ndarray(self.encoder.encode(text)) for text in texts])