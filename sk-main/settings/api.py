import json

import requests
from pymongo import MongoClient


class APIImplementation:
    def fetch_data(self, url: str):
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.text)


class DataIngestionAPIBridge:
    def __init__(
        self,
        api_connection_params: dict[str, str],
        api_implementation: APIImplementation,
        api_to_system_map: dict[str, str],
    ) -> None:
        self.api_connection_params: dict[str, str] = api_connection_params
        self.api_implementation: APIImplementation = api_implementation
        self.api_to_system_map: dict[str, str] = api_to_system_map

    def ingest_data(self):
        connection_url = self.api_connection_params.get("connection_url")
        if not connection_url:
            raise ValueError("Could not parse connection url.")
        data = self.api_implementation.fetch_data(connection_url)
        translated_data = self.translate_data(data)
        self.store_data(translated_data)

    def translate_data(self, data):
        translated_data = {}
        for key, value in self.api_to_system_map.items():
            translated_data[value] = data.get(key, None)
        return translated_data

    def store_data(self, translated_data):
        client = MongoClient("localhost", 27017)
        db = client["mydatabase"]
        db.customers.insert_one(translated_data["customer"])
        db.products.insert_many(translated_data["products"])
        db.purchases.insert_one(translated_data["purchase"])
