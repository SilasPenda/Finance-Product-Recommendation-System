import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import models as qmodels

from src.exception import CustomException
from src.logger import logging


load_dotenv()


class Upserter:
    """
    Handles batched upserts into Qdrant.
    """

    def __init__(self, client):
        self.client = client

    def _ensure_collection(self, collection_name: str, vector_size: int=384):
        """
        Create the collection if it does not already exist.
        """
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

        logging.info(f"Created collection: {collection_name}")

    def upsert(
        self,
        embeddings: Any,
        targets: List[str],
        ids: List[str],
        collection_name: str,
        batch_size: int = 100,
    ):
        """
        Upsert embeddings into Qdrant in safe, bounded batches.
        """
        try:
            embeddings = np.array(embeddings)
            vector_size = embeddings.shape[1]
            self._ensure_collection(collection_name, vector_size)

            total_points = len(ids)

            for start in range(0, total_points, batch_size):
                end = min(start + batch_size, total_points)

                points = [
                    qmodels.PointStruct(
                        id=ids[i],
                        vector=embeddings[i].tolist(),
                        payload={
                        "metadata": {
                            "subscribed": targets[i],
                            "collection_version": collection_name
                        },
                    },
                    )
                    for i in range(start, end)
                ]

                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )

            logging.info(
                f"Upserted {total_points} points into collection '{collection_name}'"
            )

        except Exception as e:
            raise CustomException(e, sys)
 