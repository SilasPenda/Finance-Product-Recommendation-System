import os
import sys
from src.logger import logging

import pytest
from dotenv import load_dotenv

from src.utils import get_llm, db_client_connect


load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
llm_type = os.getenv("LLM_TYPE")
model_name = os.getenv("MODEL_NAME")
collection_name = os.getenv("COLLECTION_NAME")


def test_llm_connection():
    """
    Test if the LLM can be instantiated and respond.
    """
    try:
        llm = get_llm(llm_type, model_name=model_name)
        response = llm.invoke("What is today's date?")

        assert response is not None, "LLM did not return a response."
        logging.info("LLM connection test passed.")

    except Exception as e:
        pytest.fail(f"Failed to connect to LLM '{llm_type}': {e}")

def test_db_connection():
    """
    Test the db_client_connect function.
    """
    client = db_client_connect(collection_name)

    assert client is not None, "Failed to connect to the collection client."
