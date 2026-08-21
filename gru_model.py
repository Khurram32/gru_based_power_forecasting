"""
GRU-based medium-term electricity-consumption forecaster.

neural-network models (ANN/FNN, with a
hidden layer processing lagged load + calendar + weather features, Ref.
[26]) fed by wavelet-decomposed input series (Ref. [6]), and elsewhere the
paper repeatedly notes (Section 4, "Conclusions") the dominant modern trend
of using recurrent/deep architectures because they "allow the taking into
account of the non-linear nature of power consumption data and finding
non-trivial dependencies in them."

A GRU (Gated Recurrent Unit) is a natural, more modern recurrent
architecture for this recipe: it directly models the sequential/lagged
structure of the monthly series (replacing the paper's hand-crafted lag
window fed to an MLP) while remaining lighter-weight than an LSTM -- a
reasonable choice given the paper's emphasis (Section 4) that "modern
methods ... do not always give better results" than simpler approaches, so
keeping the model small guards against overfitting a short monthly series.
"""

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """GRU regressor: (batch, lookback, n_features) -> (batch,) next-step
    scaled consumption forecast."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        head_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_in // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, h_n = self.gru(x)
        # Use the last time step's output (equivalent to the final hidden
        # state of the top layer for a unidirectional GRU)
        last_out = out[:, -1, :]  # (batch, hidden_size * num_directions)
        pred = self.head(last_out).squeeze(-1)  # (batch,)
        return pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape/sanity test
    torch.manual_seed(0)
    model = GRUForecaster(n_features=8, hidden_size=64, num_layers=2, dropout=0.2)
    dummy_x = torch.randn(4, 12, 8)  # batch=4, lookback=12, n_features=8
    out = model(dummy_x)
    print("Output shape:", out.shape)
    assert out.shape == (4,), f"Unexpected output shape: {out.shape}"
    print("Trainable parameters:", model.count_parameters())
    print("GRU model forward-pass sanity check passed.")
