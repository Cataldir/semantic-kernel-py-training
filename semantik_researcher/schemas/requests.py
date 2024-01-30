from __future__ import annotations

from pydantic import BaseModel


class ChatEndpoint(BaseModel):
    prompt: str
