import os
import joblib
import pytest
import pandas as pd
from dotenv import load_dotenv

from deployment.inference_pipeline import Inference
from src.utils import get_config, db_client_connect


load_dotenv()


@pytest.mark.skipif(
    not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"),
    reason="Qdrant credentials not set"
)

@pytest.mark.skipif(
    not os.getenv("COLLECTION_NAME"),
    reason="Contract collection name not set"
)

def test_qdrant_retrieval():
    inference = Inference()
    collection = os.getenv("COLLECTION_NAME")

    config = get_config(os.path.join(os.getcwd(), "config.yaml"))
    feature_columns = config["FEATURE_COLUMNS"]

    client = db_client_connect(collection)

    test_data = os.path.join(os.getcwd(), os.getenv("DATA_DIR"), "test_data.csv")
    df = pd.read_csv(test_data)
    df = df.sample(1)

    features = df[feature_columns]
    embeddings = inference.preprocess(features)

    response = client.query_points(
            collection_name=collection,
            query=embeddings[0].tolist(),
            limit=1,
            with_payload=True
    )

    points = response.points


    results = [
            {
                "id": point.id,
                "score": point.score,
                "subscribed": point.payload["metadata"]["subscribed"]
            }
            for point in points
        ]

    assert results is not None, "No retrieval results returned."
        
    # Cleanup
    # client.delete_collection(collection_name=collection)