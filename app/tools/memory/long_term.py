from __future__ import annotations

import uuid
import asyncio
from typing import Any, List, Tuple, Dict

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

from pymongo.results import DeleteResult, UpdateResult, InsertOneResult
from semantic_kernel.memory.memory_record import MemoryRecord
import tiktoken

from ._abstract import CosmosAbstractMemory
from .utils import apply_gaussian_mixture_similarity


class CosmosMongoMemory(CosmosAbstractMemory):

    async def create(self, dataclass_instance: Dict[str, Any]) -> Any:
        document: dict[str, Any] = dataclass_instance
        for key, value in document.items():
            if isinstance(value, uuid.UUID):
                document[key] = str(value)
        result: InsertOneResult = await self.database[
            dataclass_instance.__class__.__name__
        ].insert_one(document)
        return result.inserted_id

    async def read(self, collection: str, document_id: uuid.UUID):
        result = await self.database[collection].find_one({"_id": document_id})
        return result

    async def update(self, document_id: uuid.UUID, dataclass_instance: Dict[str, Any]):
        result: UpdateResult = await self.database[
            dataclass_instance.__class__.__name__
        ].update_one({"_id": document_id}, {"$set": dataclass_instance})
        return result.modified_count

    async def delete(self, collection: str, document_id: uuid.UUID):
        result: DeleteResult = await self.database[collection].delete_one({"_id": document_id})
        return result.deleted_count

    @staticmethod
    def __to_dict(memory: MemoryRecord) -> Dict[str, Any]:
        return dict(
            key=memory._key,
            timestamp=memory._timestamp,
            is_reference=memory._is_reference,
            external_source_name=memory._external_source_name,
            id=memory._id,
            description=memory._description,
            text=memory._text,
            additional_metadata=memory._additional_metadata,
            embedding=memory._embedding,
        )

    async def create_collection(self, collection_name: str) -> None:
        """Creates a new collection in the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            None
        """
        await self.database.create_collection(collection_name)

    async def get_collections(self) -> List[str]:
        """Gets all collection names in the data store.

        Returns:
            List[str] -- A group of collection names.
        """
        return await self.database.list_collection_names()

    async def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection from the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            None
        """
        await self.database.drop_collection(collection_name)

    async def does_collection_exist(self, collection_name: str) -> bool:
        """Determines if a collection exists in the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            bool -- True if given collection exists, False if not.
        """
        return self.database.get_collection(collection_name) is not None

    async def upsert(self, collection_name: str, record: MemoryRecord) -> str:
        """Upserts a memory record into the data store. Does not guarantee that the collection exists.
            If the record already exists, it will be updated.
            If the record does not exist, it will be created.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            record {MemoryRecord} -- The memory record to upsert.

        Returns:
            str -- The unique identifier for the memory record.
        """
        collection = await self.read(collection_name, uuid.UUID(record._key)) or None
        if collection:
            await self.update(uuid.UUID(record._key), self.__to_dict(record))
            item = record._key
        else:
            instance = await self.create(self.__to_dict(record))
            item = str(instance)
        return item

    async def upsert_batch(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        """Upserts a group of memory records into the data store. Does not guarantee that the collection exists.
            If the record already exists, it will be updated.
            If the record does not exist, it will be created.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            records {MemoryRecord} -- The memory records to upsert.

        Returns:
            List[str] -- The unique identifiers for the memory records.
        """
        return await asyncio.gather(*map(lambda record: self.upsert(collection_name, record), records))

    async def get(self, collection_name: str, key: str, with_embedding: bool) -> MemoryRecord:
        """Gets a memory record from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            key {str} -- The unique id associated with the memory record to get.
            with_embedding {bool} -- If true, the embedding will be returned in the memory record.

        Returns:
            MemoryRecord -- The memory record if found
        """
        item = await self.read(collection_name, uuid.UUID(key))
        if item:
            if with_embedding:
                encoder = tiktoken.get_encoding("cl100k_base")
                embedding = encoder.encode(item['text'])
            response = MemoryRecord(
                key=item['key'],
                timestamp=item['timestamp'],
                is_reference=item['is_reference'],
                external_source_name=item['external_source_name'],
                id=item['id'],
                description=item['description'],
                text=item['text'],
                additional_metadata=item['additional_metadata'],
                embedding=np.ndarray(item['embedding']) if embedding else None
            )
            return response
        else:
            raise MemoryError(f"Memory record with key {key} not found in collection {collection_name}")

    async def get_batch(self, collection_name: str, keys: List[str], with_embeddings: bool) -> List[MemoryRecord]:
        """Gets a batch of memory records from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            keys {List[str]} -- The unique ids associated with the memory records to get.
            with_embeddings {bool} -- If true, the embedding will be returned in the memory records.

        Returns:
            List[MemoryRecord] -- The memory records associated with the unique keys provided.
        """
        return await asyncio.gather(*map(lambda key: self.get(collection_name, key, with_embeddings), keys))

    async def remove(self, collection_name: str, key: str) -> None:
        """Removes a memory record from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            key {str} -- The unique id associated with the memory record to remove.

        Returns:
            None
        """
        await self.delete(collection_name, uuid.UUID(key))

    async def remove_batch(self, collection_name: str, keys: List[str]) -> None:
        """Removes a batch of memory records from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            keys {List[str]} -- The unique ids associated with the memory records to remove.

        Returns:
            None
        """
        await asyncio.gather(*map(lambda key: self.remove(collection_name, key), keys))

    async def get_nearest_match(
        self,
        collection_name: str,
        embedding: np.ndarray,
        min_relevance_score: float,
        with_embedding: bool,
    ) -> Tuple[MemoryRecord, float]:
        """Gets the nearest match to an embedding of type float. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            embedding {ndarray} -- The embedding to compare the collection's embeddings with.
            min_relevance_score {float} -- The minimum relevance threshold for returned result.
            with_embedding {bool} -- If true, the embeddings will be returned in the memory record.

        Returns:
            Tuple[MemoryRecord, float] -- A tuple consisting of the MemoryRecord and the similarity score as a float.
        """
        response = await self.get_nearest_matches(
            collection_name=collection_name,
            embedding=embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding,
        )
        return response[0]

    async def get_nearest_matches(
        self,
        collection_name: str,
        embedding: np.ndarray,
        limit: int,
        min_relevance_score: float,
        with_embeddings: bool,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Gets the nearest matches to an embedding of type float. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            embedding {ndarray} -- The embedding to compare the collection's embeddings with.
            limit {int} -- The maximum number of similarity results to return.
            min_relevance_score {float} -- The minimum relevance threshold for returned results.
            with_embeddings {bool} -- If true, the embeddings will be returned in the memory records.

        Returns:
            List[Tuple[MemoryRecord, float]] -- A list of tuples where item1 is a MemoryRecord and item2
                is its similarity score as a float.
        """
        documents = await self.database[collection_name].find({}).to_list(length=limit)
        embeddings = np.array([doc['embedding'] + embedding for doc in documents])
        clusters = apply_gaussian_mixture_similarity(embeddings)
        embeddings_clusters = {}
        for cluster, document in zip(clusters, documents):
            document['cluster'] = cluster

        for doc in documents:
            # Get the cluster of the document
            cluster = doc['cluster']

            # If the cluster is not in the dictionary yet, add it
            if cluster not in embeddings_clusters:
                clusters[cluster] = []

            # Add the document to the list of documents in its cluster
            embeddings_clusters[cluster].append(doc)

        similarities = cosine_similarity(np.array([embedding_cluster for embedding_cluster in embeddings_clusters.values()]), embeddings)[0]

        ranked_documents = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
        return [doc for doc, similarity in ranked_documents if similarity > min_relevance_score][:limit]
