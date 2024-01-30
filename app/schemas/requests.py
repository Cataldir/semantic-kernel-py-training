from __future__ import annotations

import uuid
from pydantic import BaseModel


class ChatEndpoint(BaseModel):
    prompt: str
    _id: uuid.UUID = uuid.uuid4()
    chat_name: str = 'researcher'
    max_tokens: int = 4096