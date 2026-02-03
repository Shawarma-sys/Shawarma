import torch
import torch.nn as nn


class RNNTeacher(nn.Module):
    """An RNN teacher model with the same input format as RNN1.

    Expected input:
        x: Long tensor with shape (batch, seq_len, feature_dim)
           feature 0: length token id
           feature 1: ipd token id

    Output:
        logits with shape (batch, labels_num)

    Notes:
                - Keeps the same constructor signature style as RNN1.
                - Compared to RNN1, it uses a GRU and stronger pooling (mean + max).
    """

    name = "RNNTeacher"

    def __init__(
        self,
        rnn_in,
        hidden_size,
        labels_num,
        len_vocab,
        ipd_vocab,
        len_embedding_bits,
        ipd_embedding_bits,
        device,
        droprate,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.device = device
        self.dropout = float(droprate)

        self.labels_num = int(labels_num)
        self.len_vocab = int(len_vocab)
        self.ipd_vocab = int(ipd_vocab)
        self.len_embedding_bits = int(len_embedding_bits)
        self.ipd_embedding_bits = int(ipd_embedding_bits)

        emb_dim = self.len_embedding_bits + self.ipd_embedding_bits

        # Keep capacity close to the CLI-provided dims (rnn_in/hidden_size) to
        # avoid an overly-large teacher that can overfit early.
        proj_dim = int(rnn_in)
        hidden = int(hidden_size)
        num_layers = int(num_layers)
        bidirectional = bool(bidirectional)

        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ).to(device)

        self.rnn = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        ).to(device)

        rnn_out_dim = hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(rnn_out_dim * 2, rnn_out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(rnn_out_dim, self.labels_num),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).long()

        len_ids = x[:, :, 0]
        ipd_ids = x[:, :, 1]

        len_ids = torch.clamp(len_ids, 0, self.len_vocab - 1)
        ipd_ids = torch.clamp(ipd_ids, 0, self.ipd_vocab - 1)

        len_x = self.len_embedding(len_ids)
        ipd_x = self.ipd_embedding(ipd_ids)

        h = torch.cat((len_x, ipd_x), dim=-1)  # (B, L, emb_dim)
        h = self.proj(h)  # (B, L, proj_dim)

        out, _ = self.rnn(h)  # (B, L, rnn_out_dim)

        # Stronger pooling than "last step": use both mean + max pooling.
        mean_pool = torch.mean(out, dim=1)
        max_pool = torch.amax(out, dim=1)
        feat = torch.cat([mean_pool, max_pool], dim=-1)

        return self.head(feat)
