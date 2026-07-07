import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    Acts like the stock TransformerEncoderLayer **but** also returns
    `attn_weights` from its internal Multi‑Head Attention block.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # ---- Self‑attention (+ return weights) ----
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,  # << key line
            average_attn_weights=False,  # keep heads separate
        )

        # ---- Residual + layer‑norm ----
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # ---- Feed‑forward ----
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)

        return src, attn_weights  # << extra return


class SimpleDecoderOnlyTransformer(nn.Module):
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
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 64, dropout)
        # encoder_layer = EncoderLayerWithAttn(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=64,
        #     dropout=dropout,
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layers = nn.ModuleList(
            [
                EncoderLayerWithAttn(d_model, nhead, 64, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool_query = nn.Parameter(torch.randn(d_model))
        self.fc_out = nn.Linear(d_model, num_continuous)
        self.pred_len = pred_len

    @torch.no_grad()  # remove if you back-prop through attn visualisation
    def _build_causal_mask(self, seq_len, device):
        """square subsequent mask identical to PyTorch's helper"""
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(
            device
        )

    def forward(
        self, src, *, return_attn=False, return_hidden=False, return_weights=False
    ):
        x = self.input_proj(src)  # B,T,d
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # T,B,d

        seq_len = x.size(0)
        causal_mask = self._build_causal_mask(seq_len, x.device)

        attn_collector, hidden_collector = [], []
        for layer in self.layers:  # <-- manual loop
            x, w = layer(x, src_mask=causal_mask)
            if return_attn:
                attn_collector.append(w.detach())
            if return_hidden:
                hidden_collector.append(x.detach())  # [T,B,d]

        # pred_tokens = x[-self.pred_len :].transpose(0, 1)  # B,pred_len,d
        # out = self.fc_out(pred_tokens).squeeze(1)  # B,F
        scores = torch.einsum("tbd,d->tb", x, self.pool_query)  # [T, B]
        weights = F.softmax(scores, dim=0).unsqueeze(-1)  # [T, B, 1]
        pooled = (weights * x).sum(dim=0)  # [B, d]

        # 4) Prediction head
        out = self.fc_out(pooled)  # [B, num_continuous]

        if return_attn and return_hidden and return_weights:
            return out, attn_collector, hidden_collector, weights  # ★ NEW
        if return_attn and return_hidden:
            return out, attn_collector, hidden_collector
        if return_attn and return_weights:  # ★ NEW
            return out, attn_collector, weights
        if return_hidden and return_weights:  # ★ NEW
            return out, hidden_collector, weights
        if return_attn:
            return out, attn_collector
        if return_hidden:
            return out, hidden_collector
        if return_weights:  # ★ NEW
            return out, weights
        return out


if __name__ == "__main__":
    # ----- DUMMY DATA -----
    batch_size = 4
    seq_len = 3
    num_features = 7
    pred_len = 1
    input_size = 12

    x = torch.randn(batch_size, seq_len, input_size)  # [4, 20, 7]
    y = torch.randn(batch_size, pred_len, num_features)  # [4, 1, 7]

    # ----- MODEL -----
    model = SimpleDecoderOnlyTransformer(
        input_size=input_size,
        num_continuous=num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        pred_len=pred_len,
    )

    # ----- FORWARD PASS -----
    out = model(x)

    print("Input shape:", x.shape)  # torch.Size([4, 20, 7])
    print("Target shape:", y.shape)  # torch.Size([4, 1, 7])
    print("Model output shape:", out.shape)  # Should be torch.Size([4, 1, 7])
