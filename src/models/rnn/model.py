import torch
from torch import nn


class RNNBaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, num_continuous=7):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size  # Predicting the same features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_continuous = num_continuous
        self.num_binary = input_size - self.num_continuous
        self.device = device

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc_cont = nn.Linear(self.hidden_size, self.num_continuous)

    def init_hidden(self, batch_size):
        # RNN requires (num_layers, batch, hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden.to(self.device)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        features, _ = self.rnn(
            x, hidden
        )  # out shape: (batch_size, seq_len, hidden_size)
        fc_cont = self.fc_cont(features[:, -1, :])  # Last time step
        return fc_cont


def test_rnn_base_model():
    batch_size = 8
    seq_len = 24
    input_size = 17
    hidden_size = 32
    num_layers = 1
    num_continuous = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNBaseModel(
        input_size, hidden_size, num_layers, device, num_continuous=num_continuous
    )
    model.to(device)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    dummy_input = dummy_input.to(device)
    output = model(dummy_input)

    assert output.shape == (
        batch_size,
        num_continuous,
    ), f"Expected output shape {(batch_size, num_continuous)}, got {output.shape}"
    print("✅ RNNBaseModel test passed!")


if __name__ == "__main__":
    test_rnn_base_model()
    print("All tests passed!")
