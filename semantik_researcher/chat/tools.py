from __future__ import annotations

import re
import sys
import json
import time
import logging
import string
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from langchain.schema import Document
from azure.storage.blob import BlobServiceClient, BlobClient
from langchain.document_loaders import AsyncHtmlLoader, PyMuPDFLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.utilities import BingSearchAPIWrapper

from schemas import SourceEngineSchema


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

nltk.download('stopwords')


class SourceEngine(ABC, BaseModel):

    research: SourceEngineSchema
    name: str
    description: str
    data: Optional[dict] = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.retrieve_data()

    @abstractmethod
    def retrieve_data(self) -> str:
        """
        retrieve data from source. Depends on schema.
        """

    def clean_text(self, text: Document) -> Tuple:
        """ Cleans text from special characters. """
        content = text.page_content
        cleaned_text: str = content.replace(
            '\n', ' '
        ).replace(
            '\r', ' '
        ).replace('\t', ' ')
        list_text: list[str | Any] = re.split(r'\W+', cleaned_text)
        table: dict[int, int | None] = str.maketrans(
            '', '', string.punctuation
        )
        stripped: list[str | Any] = [w.translate(table) for w in list_text]
        stop_words: list[str] = stopwords.words('english')
        cleaned_text: str = ' '.join(
            [w for w in stripped if w not in stop_words]
        )
        if 'page' in text.metadata:
            r_key = text.metadata.get('page')
        else:
            r_key = text.metadata.get('source')
        return str(r_key), cleaned_text

    def __upload_to_blob(
        self, connect_str: str, container_name: str, blob_name: str
    ) -> None:
        try:
            blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(connect_str)
            blob_client: BlobClient = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            blob_client.upload_blob(json.dumps(self.data))
        except Exception as e:
            logger.error("Error occurred while uploading to Azure Blob Storage: %s", str(e))
            raise e

    def post_data(self) -> None:
        """ Uploads data to Azure Blob Storage. """

        container_name: str = self.research.destination.get('blob-container', '')
        connect_str: str = self.research.destination.get('blob-connection-string', '')
        blob_name = f'chat-data-{time.time()}.txt'
        if any((container_name == '', connect_str == '')):
            raise AttributeError('You must define destination parameters.')
        self.__upload_to_blob(connect_str, container_name, blob_name)


class BookEngine(SourceEngine):

    name: str = 'Book Engine Tool'
    description: str = "Retrieves and clean information from book."

    def __load_book(self, book):
        pages = PyMuPDFLoader(book).load()
        book_data = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.clean_text, page) for page in pages]
            for future in concurrent.futures.as_completed(futures):
                book_data[future.result()[0]] = self.clean_text(future.result()[1])
        return book_data

    def retrieve_data(self) -> str:
        data = self.research.origin.get('path', None)
        book_data = self.__load_book(data)
        logger.info("%s Pages Summarized.", len(book_data))
        return str(book_data)


class BingEngine(SourceEngine):

    name: str = 'Bing Engine Tool'
    description: str = "Retrieves and clean information from search query."

    def __get_website_text(self, url):
        """ Extracts text from a given website URL. """
        book_data = {}
        try:
            loader = AsyncHtmlLoader(url)
            docs = loader.load()
            bs_transformer = BeautifulSoupTransformer()
            pages = bs_transformer.transform_documents(docs, tags_to_extract=["p", ])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.clean_text, page) for page in pages]
                for future in concurrent.futures.as_completed(futures):
                    book_data[future.result()[0]] = future.result()[1]
            return book_data
        except Exception as e:
            logger.error(
                "Error occurred while fetching website content: %s",
                str(e)
            )
            raise e

    def retrieve_data(self) -> Dict[str, str]:
        search = BingSearchAPIWrapper()
        query_params = self.research.origin.get('bing', None)
        topic = self.research.origin.get('topic', None)
        if query_params:
            results = search.results(
                topic,
                query_params.get('queries')
            )
            website_data = {
                key: value
                for result in results
                for key, value in self.__get_website_text(result['link']).items()
            }
            return website_data
        return {'webdata': 'None'}
