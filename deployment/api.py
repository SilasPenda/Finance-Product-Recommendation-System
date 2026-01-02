import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, request, jsonify
from deployment.inference_pipeline import Inference
from src.utils import get_config

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/product/recommend", methods=["POST"])
def process():
    try:
        inference = Inference()

        config = get_config(os.path.join(os.getcwd(), "config.yaml"))
        required_fields = config["FEATURE_COLUMNS"]

        payload = {}

        for field in required_fields:
            if field == "client_id":
                continue

            value = request.form.get(field)
            if value is None:
                return jsonify({"error": f"{field} is required"}), 400
            payload[field] = value

        # Cast numeric fields
        payload["age"] = int(payload["age"])
        payload["balance"] = float(payload["balance"])
        payload["campaign"] = int(payload["campaign"])
        payload["pdays"] = int(payload["pdays"])
        payload["previous"] = int(payload["previous"])

        # Convert to DataFrame (single-row inference)
        input_df = pd.DataFrame([payload])

        results = inference.predict(input_df)

        return jsonify({
            "client_id": request.form.get("client_id"),
            "recommendation": results
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

