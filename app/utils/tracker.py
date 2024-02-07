from typing import Callable, Any

import time
import logging
from string import Template

import tiktoken


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('performance.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def count_tokens(prompt: str) -> str:
    """
    Counts the number of tokens in the given prompt.

    Args:
        prompt (str): The prompt to count tokens for.

    Returns:
        str: The number of tokens in the prompt.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    encoder.encode(prompt)
    return str(len(prompt.split()))


def evaluate_performance(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        execution_time: float = end_time - start_time
        logger.info(
            'Execution time for %s: %s seconds',
            func.__name__,
            execution_time
        )
        return result
    return wrapper
