# Global Mobility Application Analyzer

An end-to-end MLOps pipeline that predicts **Visa Approval Status** using a trained ML model — built with a full training pipeline, a FastAPI-based prediction service, and automated CI/CD deployment to AWS EC2.

## 🏗️ Architecture

![Global Mobility Application Analyzer Architecture](architecture.png)

---

## 📌 Overview

This project is split into three major components:

1. **Training Pipeline** — ingests data from MongoDB, validates and transforms it, trains and evaluates a model, and pushes the final artifact to **Hugging Face Hub**.
2. **Prediction Pipeline** — a FastAPI web application that accepts user input, loads the latest model from Hugging Face Hub, and returns a visa approval prediction.
3. **Deployment & Infrastructure** — a CI/CD workflow that builds a Docker image from source code, pushes it to AWS ECR, and deploys it to an AWS EC2 instance.

---

## 🔁 Training Pipeline

| Stage | Description |
|---|---|
| **Data Ingestion** | Pulls raw data from MongoDB |
| **Data Validation** | Validates schema, checks for drift/missing data |
| **Data Transformation** | Cleans and transforms data for training |
| **Model Trainer** | Trains the ML model |
| **Model Evaluation** | Evaluates model performance against thresholds |
| **Model Pusher** | Pushes the trained model to **Hugging Face Hub** (Model Registry) |

## 🔮 Prediction Pipeline

| Stage | Description |
|---|---|
| **User Input (Web Form)** | User submits application details via a web form |
| **Web Application (FastAPI)** | Receives and routes the incoming request |
| **Data Transformation** | Transforms the input into model-ready format |
| **Model Loader** | Loads the latest model from **Hugging Face Hub** |
| **Prediction** | Generates the final Visa Approval Status |

## 🚀 Deployment & Infrastructure

| Stage | Tool |
|---|---|
| Source Code | GitHub |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Container Registry | AWS ECR |
| Hosting | AWS EC2 |

---

## 🗂️ Model Registry — Hugging Face Hub

Model artifacts are versioned and stored on **Hugging Face Hub** instead of AWS S3.

```python
from huggingface_hub import HfApi, hf_hub_download

# Push model
api = HfApi()
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="<your-hf-username>/global-mobility-model",
    repo_type="model",
)

# Load model
model_path = hf_hub_download(
    repo_id="<your-hf-username>/global-mobility-model",
    filename="model.pkl",
)
```

> Set your Hugging Face token as an environment variable (`HF_TOKEN`) or in GitHub Actions secrets for automated pushes during CI/CD.

---

## ⚙️ Tech Stack

- **Data Store:** MongoDB
- **Model Registry:** Hugging Face Hub
- **Backend:** FastAPI
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud:** AWS (ECR + EC2)

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/global-mobility-application-analyzer.git
cd global-mobility-application-analyzer
pip install -r requirements.txt
```

## ▶️ Running Locally

```bash
uvicorn app:app --reload
```

---

## 📄 License

This project is licensed under the MIT License.