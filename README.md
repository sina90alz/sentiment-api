# 🚀 Sentiment Analysis API (FastAPI + PyTorch + Docker)

A **production-ready sentiment analysis API** built with:

-   **PyTorch** (LSTM model)
-   **HuggingFace Tokenizer**
-   **FastAPI** (REST API + Swagger UI)
-   **Docker** (containerized deployment)

This project demonstrates a complete **end‑to‑end machine learning
pipeline**:

-   Training the model
-   Saving and loading the model
-   Serving predictions via API
-   Containerizing with Docker

------------------------------------------------------------------------

## ⭐ Features

-   📦 **Custom LSTM sentiment classifier (PyTorch)**
-   🧠 **Tokenizer from BERT (AutoTokenizer)**
-   ⚡ **FastAPI for real-time inference**
-   🐳 **Dockerized for easy deployment**
-   📝 **Automatic Swagger UI** at: `http://localhost:8000/docs`

------------------------------------------------------------------------

## 📁 Project Structure

    sentiment-api/
    │
    ├── app/
    │   ├── api.py               # FastAPI application
    │   ├── inference.py         # Prediction logic
    │   ├── model.py             # LSTM model architecture
    │   ├── train.py             # Training script
    │   ├── __init__.py
    │
    ├── tokenizer/               # Saved tokenizer
    ├── sentiment_lstm.pt        # Saved trained model
    │
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

------------------------------------------------------------------------

## 🔧 Installation

### 1️⃣ Clone repository

    git clone https://github.com/sina90alz/sentiment-api.git
    cd sentiment-api

### 2️⃣ Install dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## 🧠 Training the Model

Run the training script:

    python -m app.train

This will:

-   Download the **SST-2 dataset**

-   Train for 5 epochs

-   Save:

    -   `sentiment_lstm.pt`
    -   `tokenizer/`

------------------------------------------------------------------------

## 🔍 Running the API (Without Docker)

    uvicorn app.api:app --host 0.0.0.0 --port 8000

Swagger UI available at: ➡ **http://localhost:8000/docs**

------------------------------------------------------------------------

## 🐳 Docker Deployment

### 1️⃣ Build the image

    docker build -t sentiment-api .

### 2️⃣ Run the container

    docker run -p 8000:8000 sentiment-api

Now open: ➡ **http://localhost:8000/docs**

------------------------------------------------------------------------

## 🧩 How It Works

### 🟦 1. Tokenization

We use HuggingFace tokenizer:

-   converts text → input IDs
-   pads / truncates to length 64
-   returns PyTorch tensors

### 🟪 2. LSTM Model

Custom PyTorch model:

-   Embedding
-   LSTM (hidden_dim=128)
-   Linear layer
-   Sigmoid output

### 🟨 3. Inference

Model predicts a score between 0 and 1:

-   ≥ 0.5 → **positive**
-   \< 0.5 → **negative**

### 🟥 4. FastAPI

Provides endpoint:

    POST /predict

with Swagger docs.

------------------------------------------------------------------------

## 📈 Future Improvements

-   Replace LSTM with **DistilBERT** for higher accuracy
-   Add **metrics endpoint**
-   Add **request logging**
-   Create a **React UI** frontend
-   Deploy to **Render / AWS / Azure / GCP**

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

------------------------------------------------------------------------

## 📜 License

MIT License.

------------------------------------------------------------------------

## ⭐ Give it a Star!

If you like this project, please give it a ⭐ on GitHub!
