"""
The configuration for the web api.
"""
import json
from base64 import encode
from typing import Any, Type, Union

import jwt
import polars as pl

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RESPONSES, BodyMessage


tags_metadata: list[dict] = [
    {
        "name": "Copilot API",
        "description": """
        Provides in-place chat functionalities and data retrieval.
        """,
    }
]

description: str = """
    A web API to serve real-time chat completion on Azure Open AI.
"""

app: FastAPI = FastAPI(
    title="Copilot API",
    version="Alpha",
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
    request: Request, exc: RequestValidationError
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
        detail={"invalid-params": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.get("/connection-data")
async def connection_data(request: Request) -> JSONResponse:
    """
    load_data loads the data into the Context
    """
    request_data: bytes = await request.body()
    data: pl.DataFrame = 
    response: JSONResponse = JSONResponse({"result": data.write_json()})
    return response
