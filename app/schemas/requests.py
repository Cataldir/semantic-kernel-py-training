from __future__ import annotations
from typing import Optional

import uuid
from pydantic import BaseModel
from .agents import ChatSchema, TextSchema, AzureSearchMemorySchema


class ChatEndpoint(BaseModel):
    prompt: str
    _id: uuid.UUID = uuid.uuid4()
    chat_name: str = 'researcher'
    max_tokens: int = 4096
    search_connections: AzureSearchMemorySchema = AzureSearchMemorySchema()
    include_embeddings: Optional[bool] = False
    chat_connections: Optional[ChatSchema] = None
    text_connections: Optional[TextSchema] = None
