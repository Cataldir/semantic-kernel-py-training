from __future__ import annotations
from multiprocessing import connection

import uuid
from pydantic import BaseModel


class ChatEndpoint(BaseModel):
    prompt: str
    _id: uuid.UUID = uuid.uuid4()
    chat_name: str = 'researcher'
    max_tokens: int = 4096


class ChatEndpointWithMemory(BaseModel):
    prompt: str
    connection_string: str
    _id: uuid.UUID = uuid.uuid4()
    chat_name: str = 'researcher'
    max_tokens: int = 4096
