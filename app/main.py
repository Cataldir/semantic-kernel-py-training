"""
The configuration for the web api.
"""
from msilib import schema
from typing import List
from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding, AzureChatCompletion

from app.schemas import RESPONSES, BodyMessage, ChatEndpoint, ChatSchema, TextSchema
from app.settings.types import COMPLETION_TYPE
from app.patterns.simple import SimpleRAG, OneShotRAG, FewShotRAG
from app.tools.memory.short_term import AISearchMemory
from app.bg_tasks import load_data


tags_metadata: list[dict] = [
    {
        "name": "Semantic Kernell Study API",
        "description": """
        Provides in-depth explanation, examples and capabilitites to work with Microsoft's semantic kernel.
        """,
    }
]

description: str = """
    A web API to serve as a web app for Semantic Kernell testing and understanding.
"""

app: FastAPI = FastAPI(
    title="Semantic Kernell Study API",
    version="0.0.1a",
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """
    response_body: BodyMessage = BodyMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        data={"invalid-params": list(exc.errors())},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.post("/simple-rag/")
async def simple_rag_chat(
    prompt: ChatEndpoint,
    bg_tasks: BackgroundTasks
) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = SimpleRAG(chat_id=prompt._id)
    chat_completion = prompt.chat_connections or ChatSchema()
    chat_completion = chat_completion.model_dump()

    text_completion = prompt.text_connections or TextSchema()
    text_completion = text_completion.model_dump()

    search_schema = prompt.search_connections

    chat_completion.update({'service_id': f"chat-{str(agent._id)}"})
    text_completion.update({'service_id': f"embeddings-{str(agent._id)}"})

    completions: List[COMPLETION_TYPE] = [AzureChatCompletion(**chat_completion), ]
    embedding_gen = AzureTextEmbedding(**text_completion)
    ai_memory = AISearchMemory(vector_size=1536, field_schema=search_schema)
    config_kwargs = {"short_term_memory": ai_memory(gen_embeddings=embedding_gen)}

    response = await agent(
        completions=completions,
        prompt=prompt.prompt,
        prompt_kwargs={"max_tokens": prompt.max_tokens, 'query': prompt.prompt},
        config_kwargs=config_kwargs
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )


@app.post("/one-shot-rag/")
async def one_shot_rag_chat(
    prompt: ChatEndpoint,
    bg_tasks: BackgroundTasks
) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = OneShotRAG(chat_id=prompt._id)
    chat_completion = prompt.chat_connections or ChatSchema()
    chat_completion = chat_completion.model_dump()

    text_completion = prompt.text_connections or TextSchema()
    text_completion = text_completion.model_dump()

    search_schema = prompt.search_connections.model_dump()

    completions: List[COMPLETION_TYPE] = [AzureChatCompletion(**chat_completion), ]
    if prompt.include_embeddings:
        completions.append(AzureTextEmbedding(**text_completion))
    response = await agent(
        completions=completions,
        prompt=prompt.prompt,
        prompt_kwargs={"max_tokens": prompt.max_tokens}
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )


@app.post("/few-shot-rag/")
async def few_shot_rag_chat(
    prompt: ChatEndpoint,
    bg_tasks: BackgroundTasks
) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = FewShotRAG(chat_id=prompt._id)
    chat_completion = prompt.chat_connections or ChatSchema()
    chat_completion = chat_completion.model_dump()

    text_completion = prompt.text_connections or TextSchema()
    text_completion = text_completion.model_dump()

    search_schema = prompt.search_connections.model_dump()

    completions: List[COMPLETION_TYPE] = [AzureChatCompletion(**chat_completion), ]
    if prompt.include_embeddings:
        completions.append(AzureTextEmbedding(**text_completion))
    response = await agent(
        completions=completions,
        prompt=prompt.prompt,
        prompt_kwargs={"max_tokens": prompt.max_tokens}
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )


@app.post("/multiplexor-rag/")
async def chat_with_multiplexor_rag(
    prompt: ChatEndpoint,
    bg_tasks: BackgroundTasks
) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = SimpleRAG(chat_id=prompt._id)
    chat_completion = prompt.chat_connections or ChatSchema()
    chat_completion = chat_completion.model_dump()

    text_completion = prompt.text_connections or TextSchema()
    text_completion = text_completion.model_dump()

    search_schema = prompt.search_connections.model_dump()

    completions: List[COMPLETION_TYPE] = [AzureChatCompletion(**chat_completion), ]
    if prompt.include_embeddings:
        completions.append(AzureTextEmbedding(**text_completion))
    response = await agent(
        completions=completions,
        prompt=prompt.prompt,
        prompt_kwargs={"max_tokens": prompt.max_tokens}
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )


@app.post("/agent-swarm/")
async def chat_with_agent_swarm(
    prompt: ChatEndpoint,
    bg_tasks: BackgroundTasks
) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = SimpleRAG(chat_id=prompt._id)
    chat_completion = prompt.chat_connections or ChatSchema()
    chat_completion = chat_completion.model_dump()

    text_completion = prompt.text_connections or TextSchema()
    text_completion = text_completion.model_dump()

    search_schema = prompt.search_connections.model_dump()

    completions: List[COMPLETION_TYPE] = [AzureChatCompletion(**chat_completion), ]
    if prompt.include_embeddings:
        completions.append(AzureTextEmbedding(**text_completion))
    response = await agent(
        completions=completions,
        prompt=prompt.prompt,
        prompt_kwargs={"max_tokens": prompt.max_tokens}
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )
