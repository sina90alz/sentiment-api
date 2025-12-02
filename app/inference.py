import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import LSTMSentimentModel


class SentimentInference:
    def __init__(self, model_path="./sentiment_lstm.pt", tokenizer_path="./tokenizer"):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load model
        vocab_size = self.tokenizer.vocab_size
        self.model = LSTMSentimentModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Sigmoid only for inference (since training used BCEWithLogitsLoss)
        self.sigmoid = nn.Sigmoid()

    def predict(self, text: str):
        # Tokenize input
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            prob = self.sigmoid(logits).item()

        # Convert to label
        label = "positive" if prob >= 0.5 else "negative"

        return {
            "label": label,
            "probability": float(prob)
        }