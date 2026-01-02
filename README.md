# Finance Product Recommendation System

An AI-powered Client Product Recommendation System designed to predict whether a client will subscribe to a financial product (e.g., a term deposit) based on their profile and historical data. This system leverages feature embeddings, vector search, and ML inference pipelines to provide personalized product recommendations.

---

## Features
- **Client Profile Analysis:** Processes client data including age, balance, campaign interaction, previous product history, and more.
- **Hybrid Recommendation:** Combines classic ML preprocessing (numeric + categorical features) with embedding-based similarity search for improved recommendations.
- **Personalized Prediction:** Predicts subscription likelihood with a clear Yes / No verdict.
- **Evaluation & Metrics:** Outputs evaluation metrics (accuracy, precision, recall, F1-score) and confusion matrices to monitor model performance.
- **Extensible Pipeline:** Modular preprocessing, inference, and upsert logic for easy experimentation and scaling.
- **API Ready:** Flask API for serving recommendations in real-time.
- **Dockerized:** Ready for containerized deployment.

---

## Architecture
- **Preprocessing Pipeline:** Handles numeric scaling, categorical encoding, and binary feature mapping.
- **Embedding Generation:** Converts client profiles into vector embeddings for similarity search and ML inference.
- **Vector Search Database:** Uses a vector database (e.g., Qdrant) to store historical client embeddings for fast similarity queries.
- **Inference Pipeline:** Generates predictions by combining ML models with nearest-neighbor matching in the vector database.
- **Flask API / Frontend:** Provides endpoints for real-time subscription predictions and batch evaluation.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Vector database setup (e.g., Qdrant)
  
### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SilasPenda/Finance_Product_Recommendation_System
   cd Finance_Product_Recommendation_System

2. Create & activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate (Linux & Mac)
   ./.venv/Scripts/activate (Windows)
   
3. Install requirements:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt

4. Create .env and config.yaml files

6. Start App

   ```bash
   python deployment/api.py
