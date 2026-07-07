import torch
from torch import nn


class LSTMBaseModel(nn.Module):
    """
    Same signature as your RNNBaseModel, but uses an LSTM.
    Outputs only the continuous features (num_continuous) from
    the last time step.
    """

    def __init__(self, input_size, hidden_size, num_layers, device, num_continuous=7):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size  # predicting same features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_continuous = num_continuous
        self.num_binary = input_size - num_continuous
        self.device = device

        # ── Core recurrent block ──────────────────────────────
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Linear head for continuous outputs only
        self.fc_cont = nn.Linear(self.hidden_size, self.num_continuous)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def init_hidden(self, batch_size):
        # LSTM needs *two* tensors: (h_0, c_0) each shaped
        # (num_layers, batch, hidden_size)
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=self.device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_size)
        hidden: tuple((num_layers, batch, hidden_size),
                      (num_layers, batch, hidden_size)) or None
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # features: (batch, seq_len, hidden_size)
        # hidden_out: ((h_n, c_n) but we ignore them here)
        features, _ = self.rnn(x, hidden)

        # Use last timestep’s hidden state for regression
        fc_cont = self.fc_cont(features[:, -1, :])
        return fc_cont


# ------------------------------------------------------------------
# Quick sanity test (same as yours)
# ------------------------------------------------------------------
def test_lstm_base_model():
    batch_size = 8
    seq_len = 24
    input_size = 17
    hidden_size = 32
    num_layers = 1
    num_continuous = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMBaseModel(
        input_size, hidden_size, num_layers, device, num_continuous=num_continuous
    ).to(device)

    dummy_input = torch.randn(batch_size, seq_len, input_size, device=device)
    output = model(dummy_input)

    assert output.shape == (
        batch_size,
        num_continuous,
    ), f"Expected {(batch_size, num_continuous)}, got {output.shape}"
    print("✅ LSTMBaseModel test passed!")


if __name__ == "__main__":
    test_lstm_base_model()
    print("All tests passed!")
