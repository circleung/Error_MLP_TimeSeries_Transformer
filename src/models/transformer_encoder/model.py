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


class SimpleEncoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        num_continuous,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pred_len=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 64, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, num_continuous)
        self.pred_len = pred_len

    def forward(self, src):
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        # seq_len = x.size(0)
        # Use causal mask to prevent peeking into the future
        # mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x)  # [seq_len, batch, d_model]
        pred_tokens = out[-self.pred_len :]  # [pred_len, batch, d_model]
        pred_tokens = pred_tokens.transpose(0, 1)  # [batch, pred_len, d_model]
        out = self.fc_out(pred_tokens)  # [batch, pred_len, num_features]
        return out.squeeze(
            1
        )  # [batch, num_features] Warning: This will return the last prediction only, adjust as needed for multiple predictions


if __name__ == "__main__":
    # ----- DUMMY DATA -----
    batch_size = 4
    seq_len = 20
    num_features = 7
    pred_len = 1

    x = torch.randn(batch_size, seq_len, num_features)  # [4, 20, 7]
    y = torch.randn(batch_size, pred_len, num_features)  # [4, 1, 7]

    # ----- MODEL -----
    model = SimpleEncoderOnlyTransformer(
        num_features=num_features, d_model=32, nhead=4, num_layers=2, pred_len=pred_len
    )

    # ----- FORWARD PASS -----
    out = model(x)

    print("Input shape:", x.shape)  # torch.Size([4, 20, 7])
    print("Target shape:", y.shape)  # torch.Size([4, 1, 7])
    print("Model output shape:", out.shape)  # Should be torch.Size([4, 1, 7])
