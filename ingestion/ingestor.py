import os
import sys
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv

from ingestion.preprocessing import Preprocessor
from ingestion.upsert import Upserter
from src.exception import CustomException
from src.utils import get_next_collection_name, get_config\


load_dotenv()


class DataIngestor:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        base_name = os.getenv("COLLECTION_BASENAME")

        config = get_config(os.path.join(os.getcwd(), "config.yaml"))

        self.target = config["TARGET"]
        
        # Qdrant setup
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
            )
        
        self.collection_name = get_next_collection_name(self.client, base_name)
        self.data_dir = os.path.join(os.getcwd(), os.getenv("DATA_DIR"))
        self.preprocessor = Preprocessor()
        self.upserter = Upserter(self.client)

    def run_pipeline(self):
        try:
            embeddings = []
            ids_list = []
            targets = []

            for root, dirs, files in os.walk(self.data_dir):
                for file in tqdm(files, total=len(files), desc="Processing csvs"):
                    if file.endswith((".csv", ".CSV")) and file != "test_data.csv":
                        csv_path = os.path.join(root, file)
                        df = pd.read_csv(csv_path)
                        
                        X = df.drop(columns=[self.target])
                        # y = df[self.target].map({"yes": 1, "no": 0})
                        y = df[self.target]
                        ids = df["client_id"]

                        X_embed = self.preprocessor.preprocess(X)
                        embeddings.extend(X_embed.tolist())
                        targets.extend(y.tolist())
                        ids_list.extend(ids)

            self.upserter.upsert(embeddings, targets, ids_list, self.collection_name)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    data_ingestor = DataIngestor()
    data_ingestor.run_pipeline()