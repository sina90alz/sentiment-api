import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        # embeddings: (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # ---- Use attention_mask to compute sequence lengths ----
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
        else:
            # fallback: use full length
            lengths = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long)

        # pack padded sequence so LSTM ignores PAD tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False  # we do NOT sort the batch
        )

        _, (h, _) = self.lstm(packed)

        # h: (num_layers * num_directions, batch, hidden_dim)
        h_last = h[-1]

        # return raw logits (NO sigmoid here!)
        return self.fc(h_last)
