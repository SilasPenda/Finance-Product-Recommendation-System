import os
import sys
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from dotenv import load_dotenv
from evaluation.validate import validate_pred
from deployment.inference_pipeline import Inference

from src.utils import get_config

load_dotenv()


test_prompts = {
    "test_1": "Does this contract have a termination clause?",
    "test_2": "Is this contract compliant with GDPR laws?",
}

def run_evaluation():
    config = get_config(os.path.join(os.getcwd(), "config.yaml"))
    feature_columns = config["FEATURE_COLUMNS"]

    test_data = os.path.join(os.getcwd(), os.getenv("DATA_DIR"), "test_data.csv")
    df = pd.read_csv(test_data)
    # df = df.sample(10)  # for quick testing

    inference = Inference()

    # Drop target once
    X = df[feature_columns]
    # X = df.drop(columns=[target]).reset_index(drop=True)

    # Prepare a new column for predictions
    pred_results = []

    for i in tqdm(range(len(X)), total=len(X)):
        row = X.iloc[[i]]
        output = inference.predict(row)
        result = output[0]["subscribed"]
        score = output[0]["score"]
        pred_results.append(result)

    df = df.reset_index(drop=True)
    df["pred_subscription"] = pred_results
    df["score"] = score

    return df




if __name__ == "__main__":
    eval_dir = os.path.join(os.getcwd(), os.getenv("EVALUATION_DIR"))
    os.makedirs(eval_dir, exist_ok=True)

    save_name = f"experiment_{len(os.listdir(eval_dir))}"
    save_dir = os.path.join(eval_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    pred_df = run_evaluation()
    # pred_df.to_csv("pred.csv", index=False)

    metrics = validate_pred(pred_df, save_dir)
    print(metrics)
