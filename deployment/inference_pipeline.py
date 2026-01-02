import os
import sys
import joblib
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from src.utils import get_config, db_client_connect


load_dotenv()


class Inference:
    def __init__(self):
        config = get_config(os.path.join(os.getcwd(), "config.yaml"))

        self.numeric_features = config["NUMERIC_FEATURES"]
        self.categorical_features = config["CATEGORICAL_FEATURES"]
        self.collection = os.getenv("COLLECTION_NAME")

        self.pipeline = joblib.load(os.getenv("DATA_TRANSFORMER"))
    
    def preprocess(self, features):
        features = features.copy()

        embeddings = self.pipeline.transform(features)

        return embeddings

    def find_matching_points(self, embeddings, top_k=1):
        """
        Tool to find matching policies based on a query using embeddings.
        Args:
            query (str): The query to find matching policies for.
            top_k (int): The number of top matching policies to return.
        """

        client = db_client_connect(self.collection)

        response = client.query_points(
            collection_name=self.collection,
            query=embeddings[0].tolist(),
            limit=top_k,
            with_payload=True
        )

        points = response.points

        return [
            {
                "id": point.id,
                "score": point.score,
                "subscribed": point.payload["metadata"]["subscribed"]
            }
            for point in points
        ]
    
    def predict(self, features):
        embeddings = self.preprocess(features)

        results = self.find_matching_points(embeddings)

        return results

