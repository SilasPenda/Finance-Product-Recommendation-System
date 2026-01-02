import os
import sys
import joblib
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import get_config

load_dotenv()


class Preprocessor:
    def __init__(self):
        self.data_transformers_dir = os.path.join(os.getcwd(), "artifacts", "data_transformers")
        os.makedirs(self.data_transformers_dir, exist_ok=True)

        config = get_config(os.path.join(os.getcwd(), "config.yaml"))
        self.numeric_features = config["NUMERIC_FEATURES"]
        self.categorical_features = config["CATEGORICAL_FEATURES"]

        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ])

        self.pipeline = Pipeline(steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, self.numeric_features),
                        ("cat", categorical_transformer, self.categorical_features),
                    ]
                )
            )
        ])

    def preprocess(self, features):
        """
        Fit preprocessing pipeline on training data
        and return embeddings
        """
        embeddings = self.pipeline.fit_transform(features)
        
        new_pipeline_name = f"preprocess_pipeline_v{len(os.listdir(self.data_transformers_dir)) + 1}.joblib"

        joblib.dump(
            self.pipeline,
            os.path.join(self.data_transformers_dir, new_pipeline_name)
        )

        return embeddings