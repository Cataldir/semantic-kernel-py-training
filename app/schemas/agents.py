from __future__ import annotations

import os
from os.path import abspath, dirname, join
from typing import Dict, Any

from pydantic import BaseModel
from dotenv import load_dotenv


dotenv_path = join(dirname(dirname(abspath(__file__))), '.env')
load_dotenv(dotenv_path)


class ChatSchema(BaseModel):
    deployment_name: str = os.getenv('AZURE_OPENAI_FOURTH_DEPLOYMENT_NAME', '')
    api_key: str = os.getenv('AZURE_OPENAI_FOURTH_API_KEY', '')
    endpoint: str = os.getenv('AZURE_OPENAI_FOURTH_ENDPOINT', '')


class TextSchema(BaseModel):
    deployment_name: str = os.getenv('AZURE_OPENAI_THIRD_DEPLOYMENT_NAME', '')
    api_key: str = os.getenv('AZURE_OPENAI_THIRD_API_KEY', '')
    endpoint: str = os.getenv('AZURE_OPENAI_THIRD_ENDPOINT', '')


class SourceEngineSchema(BaseModel):
    origin: Dict[str, Any]
    destination: Dict[str, Any]


class SearchEngineSchema(BaseModel):
    endpoint: str = os.environ.get('AZURE_AI_SEARCH_SERVICE', '')
    index_name: str = os.environ.get('AZURE_AI_INDEX', '')
    search_key: str = os.environ.get('AZURE_AI_SEARCH_KEY', '')
