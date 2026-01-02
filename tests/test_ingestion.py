import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingestion.upsert import Upserter
from ingestion.preprocessing import Preprocessor
from src.utils import get_config


load_dotenv()


# Fixture to provide chunked contract
@pytest.fixture
def preprocess():
    preprocessor = Preprocessor()

    csv_path = os.path.join(os.getcwd(), "artifacts/data_balanced/test_data.csv")

    config = get_config(os.path.join(os.getcwd(), "config.yaml"))
    target = config["TARGET"]

    embeddings = []
    ids_list = []
    targets = []

    df = pd.read_csv(csv_path)
    df = df.sample(10)
                        
    X = df.drop(columns=[target])
    y = df[target].map({"yes": 1, "no": 0})
    ids = df["client_id"]

    X_embed = preprocessor.preprocess(X)
    embeddings.extend(X_embed)
    targets.extend(y.tolist())
    ids_list.extend(ids)

    assert embeddings, "Failed to generate embeddings."
    assert targets, "Chunking failed to extract target values."

    return embeddings, targets, ids_list


# Chunking test
def test_chunking(preprocess):
    embeddings, targets, ids_list = preprocess


@pytest.mark.skipif(
    not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"),
    reason="Qdrant credentials not set"
)
@pytest.mark.skipif(
    not os.getenv("TEST_COLLECTION_NAME"),
    reason="Test collection name not set"
)


def test_ingestion(preprocess):
    embeddings, targets, ids_list = preprocess

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")


    # Qdrant setup
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
        )
    
    collection_name = os.getenv("TEST_COLLECTION_NAME")

    upserter = Upserter(client)
    upserter.upsert(embeddings, targets, ids_list, collection_name)

