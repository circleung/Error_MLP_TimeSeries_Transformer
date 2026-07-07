import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1)].to(x.device)


class SimpleTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pred_len=1,
        output_dim=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=64, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_dim = output_dim or num_features  # Predict all features by default
        self.fc_out = nn.Linear(
            d_model, self.output_dim * pred_len
        )  # Output as flattened vector
        self.pred_len = pred_len

    def forward(self, src):
        """
        src: [batch_size, seq_len, num_features]
        Returns: [batch_size, pred_len * output_dim]
        """
        x = self.input_proj(src)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # Add positional encoding
        x = x.transpose(
            0, 1
        )  # [seq_len, batch_size, d_model] (PyTorch transformer expects this)
        x = self.transformer_encoder(x)
        x = x[-1]  # Use last token (or pool over time) [batch_size, d_model]
        out = self.fc_out(x)  # [batch_size, pred_len * output_dim]
        out = out.view(
            -1, self.pred_len, self.output_dim
        )  # [batch_size, pred_len, output_dim]
        return out  # Returns (batch_size, pred_len, output_dim)


if __name__ == "__main__":
    # ----- DUMMY DATA -----
    batch_size = 4
    seq_len = 20
    num_features = 17
    pred_len = 1

    x = torch.randn(batch_size, seq_len, num_features)  # [4, 20, 7]
    y = torch.randn(batch_size, pred_len, num_features)  # [4, 1, 7]

    # ----- MODEL -----
    model = SimpleTimeSeriesTransformer(
        num_features=num_features, d_model=32, nhead=4, num_layers=2, pred_len=pred_len
    )

    # ----- FORWARD PASS -----
    out = model(x)

    print("Input shape:", x.shape)  # torch.Size([4, 20, 7])
    print("Target shape:", y.shape)  # torch.Size([4, 1, 7])
    print("Model output shape:", out.shape)  # Should be torch.Size([4, 1, 7])
