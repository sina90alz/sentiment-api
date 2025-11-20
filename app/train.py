import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset
import os

from .model import LSTMSentimentModel  # import your model

# -------------------------------
# Custom Dataset
# -------------------------------
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# -------------------------------
# Training Function
# -------------------------------
def train():
    # 1️⃣ Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # 2️⃣ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3️⃣ Tokenize dataset
    train_texts = [str(s) for s in dataset["train"]["sentence"]]
    val_texts = [str(s) for s in dataset["validation"]["sentence"]]

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    # 4️⃣ Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 5️⃣ Initialize model (match your model.py)
    vocab_size = tokenizer.vocab_size
    model = LSTMSentimentModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6️⃣ Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 7️⃣ Save model and tokenizer
    os.makedirs("./tokenizer", exist_ok=True)
    torch.save(model.state_dict(), "./sentiment_lstm.pt")
    tokenizer.save_pretrained("./tokenizer")
    print("Model and tokenizer saved!")

# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    train()