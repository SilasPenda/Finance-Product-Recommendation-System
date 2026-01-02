import os
import re
import sys
import yaml
import torch
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.exception import CustomException


load_dotenv()

# Get the absolute path to the root of the project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add src to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["OPENAI_API_KEY"]
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


def read_yaml(file_path):
    """
    Reads a YAML file and returns the content as a Python dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """

    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        return content
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None


def get_device():
    """
    Get the device to be used for tensor operations.
    Returns:
        device: The device to be used (CPU, CUDA, or MPS).
    """
    # if torch.backends.mps.is_available():
    #     return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_llm(type="openai", model_name="gpt-4o"):
    """
    Get a language model based on the specified type and model name.
    Args:
        type (str): The type of language model to use ("openai" or "ollama").
        model_name (str): The name of the model to use.
    Returns:
        llm: An instance of the specified language model.
    """

    llms = {
        "openai": ChatOpenAI(model_name=model_name, openai_api_key = openai_api_key),
        "ollama": OllamaLLM(model=model_name)
    }
    llm = llms[type]

    return llm


def db_client_connect(collection_name: str, vector_size: int = 384):
    """
    Connect to a client collection.

    Args:
        collection_name (str): Name of the collection.
        url (str, optional): Qdrant Cloud URL.
        api_key (str, optional): Qdrant Cloud API key.
        host (str): Local Qdrant host.
        port (int): Local Qdrant port.
        vector_size (int): Embedding dimension.

    Returns:
        QdrantClient: Connected client with ensured collection.
    """

    try:
        # Cloud or local connection
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        return client

    except Exception as e:
        print(f"Error getting Qdrant collection '{collection_name}': {e}")
        return None

def get_config(config_filepath: str) -> dict:
    try:
        with open(config_filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}
    

def get_latest_collection_version(client, base_name: str) -> int:
    """
    Returns the highest version number for collections named base_name_vN.
    If none exist, returns 0.
    """
    pattern = re.compile(rf"^{base_name}_v(\d+)$")

    versions = []

    collections = client.get_collections().collections
    for c in collections:
        match = pattern.match(c.name)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) if versions else 0

def get_next_collection_name(client, base_name: str) -> str:
    latest_version = get_latest_collection_version(client, base_name)
    next_version = latest_version + 1
    return f"{base_name}_v{next_version}"

