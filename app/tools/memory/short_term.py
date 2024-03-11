import logging
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)


logger: logging.Logger = logging.getLogger(__name__)



class AISearchMemory:
    acs_memory_store = AzureCognitiveSearchMemoryStore(
        vector_size=1536,
        search_endpoint=azure_ai_search_url,
        admin_key=azure_ai_search_api_key,
    )

    memory = SemanticTextMemory(storage=acs_memory_store, embeddings_generator=embedding_gen)
