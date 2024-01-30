"""
A package that manages the response bodies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_202_ACCEPTED,
    HTTP_301_MOVED_PERMANENTLY,
    HTTP_302_FOUND,
    HTTP_307_TEMPORARY_REDIRECT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_418_IM_A_TEAPOT,
    HTTP_422_UNPROCESSABLE_ENTITY,
)


@dataclass
class BodyMessage:
    """
    The base message body for HTTP responses
    """

    success: bool
    type: Optional[str]
    title: Optional[str]
    data: Optional[
        Union[
            Dict[str, Union[str, List]], List[Dict[str, Union[str, List]]]
        ]
    ]


RESPONSES = {
    HTTP_200_OK: {"model": BodyMessage},
    HTTP_201_CREATED: {"model": BodyMessage},
    HTTP_202_ACCEPTED: {"model": BodyMessage},
    HTTP_302_FOUND: {"model": BodyMessage},
    HTTP_301_MOVED_PERMANENTLY: {"model": BodyMessage},
    HTTP_307_TEMPORARY_REDIRECT: {"model": BodyMessage},
    HTTP_400_BAD_REQUEST: {"model": BodyMessage},
    HTTP_401_UNAUTHORIZED: {"model": BodyMessage},
    HTTP_403_FORBIDDEN: {"model": BodyMessage},
    HTTP_418_IM_A_TEAPOT: {"model": BodyMessage},
    HTTP_422_UNPROCESSABLE_ENTITY: {"model": BodyMessage},
}
