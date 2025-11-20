import torch
import torch.nn as nn

class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        _, (h, _) = self.lstm(x)

        out = self.fc(h[-1])
        return self.sigmoid(out)
