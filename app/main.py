"""
The configuration for the web api.
"""
from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import RESPONSES, BodyMessage, ChatEndpoint
from app.research import Researcher
from app.bg_tasks import load_data


tags_metadata: list[dict] = [
    {
        "name": "Copilot API",
        "description": """
        Provides in-place chat functionalities and data retrieval.
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
        data={"invalid-params": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.post("/chat-with-history")
async def connection_data(prompt: ChatEndpoint, bg_tasks: BackgroundTasks) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    agent = Researcher(chat_id=prompt._id)
    response = await agent(
        chat_name=prompt.chat_name,
        prompt=prompt.prompt,
        max_tokens=prompt.max_tokens
    )
    bg_tasks.add_task(load_data, response)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )
