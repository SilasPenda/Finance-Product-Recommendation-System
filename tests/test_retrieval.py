import os
import pytest
import pandas as pd
from dotenv import load_dotenv

from deployment.inference_pipeline import Inference
from src.utils import get_config


load_dotenv()


@pytest.mark.skipif(
    not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"),
    reason="Qdrant credentials not set"
)

@pytest.mark.skipif(
    not os.getenv("TEST_COLLECTION_NAME"),
    reason="Contract collection name not set"
)

def test_qdrant_retrieval():
    inference = Inference()

    config = get_config(os.path.join(os.getcwd(), "config.yaml"))
    feature_columns = config["FEATURE_COLUMNS"]

    test_data = os.path.join(os.getcwd(), os.getenv("DATA_DIR"), "test_data.csv")
    df = pd.read_csv(test_data)
    df = df.sample(1)

    X = df[feature_columns]
    results = inference.predict(X)

    assert results is not None, "No retrieval results returned."
        
