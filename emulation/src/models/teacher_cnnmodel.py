import torch
import torch.nn as nn


class TextCNNTeacher(nn.Module):
    """A stronger teacher model with the same input format as TextCNN1.

    Expected input:
        x: Long tensor with shape (batch, seq_len, feature_dim)
           feature 0: length token id
           feature 1: ipd token id (optional; if missing will be treated as zeros)

    Output:
        logits with shape (batch, num_classes)
    """

    name = "TextCNNTeacher"

    def __init__(
        self,
        input_size,
        num_classes,
        len_vocab,
        ipd_vocab,
        len_embedding_bits,
        ipd_embedding_bits,
        nk,
        ebdin,
        device,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.len_vocab = int(len_vocab)
        self.ipd_vocab = int(ipd_vocab)
        self.len_embedding_bits = int(len_embedding_bits)
        self.ipd_embedding_bits = int(ipd_embedding_bits)
        self.device = device

        emb_dim = self.len_embedding_bits + self.ipd_embedding_bits

        # Make the teacher wider than the student while preserving the same constructor.
        d_model = max(32, int(ebdin) * 8)
        channels = max(64, int(nk) * 16)

        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(device)

        # Multi-kernel temporal CNN + global max pooling (length-agnostic).
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, channels, kernel_size=k, padding=0),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                ).to(device)
                for k in (3, 4, 5)
            ]
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * 3, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).long()

        len_ids = x[:, :, 0]
        if x.size(-1) > 1:
            ipd_ids = x[:, :, 1]
        else:
            ipd_ids = torch.zeros_like(len_ids)

        len_ids = torch.clamp(len_ids, 0, self.len_vocab - 1)
        ipd_ids = torch.clamp(ipd_ids, 0, self.ipd_vocab - 1)

        len_x = self.len_embedding(len_ids)
        ipd_x = self.ipd_embedding(ipd_ids)

        h = torch.cat((len_x, ipd_x), dim=-1)  # (B, L, emb_dim)
        h = self.proj(h)  # (B, L, d_model)
        h = h.transpose(1, 2)  # (B, d_model, L)

        pooled = []
        for block in self.conv_blocks:
            y = block(h)  # (B, C, L')
            y = torch.amax(y, dim=-1)  # global max over time -> (B, C)
            pooled.append(y)

        z = torch.cat(pooled, dim=-1)  # (B, 3C)
        logits = self.head(z)  # (B, num_classes)
        return logits
