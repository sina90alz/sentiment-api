import torch
from transformers import AutoTokenizer
from model import LSTMSentimentModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = LSTMSentimentModel(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("./sentiment_lstm.pt", map_location=torch.device('cpu')))
model.eval()

def predict(text):
    # Tokenize input
    tokens = tokenizer(
        [text],  # must be a list of strings for batch processing
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(tokens["input_ids"])  # LSTM model does not use attention_mask

    confidence = output.item()
    sentiment = "positive" if confidence >= 0.5 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }