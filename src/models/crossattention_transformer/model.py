"""
cross_attention_transformer.py
──────────────────────────────
* Predicts only the continuous variables (7) from a mixed-type
  multivariate sequence (7 continuous + 10 binary = 17 total).
* Encoder consumes the binary stream, decoder consumes the
  continuous stream, joined with cross-attention.
* Tiny model (d_model = 24, nhead = 3) + additive sinusoidal
  positional encoding.
* End of file contains a smoke-test / demo run.

Torch ≥ 2.1 recommended.  The “odd num_heads” warning that shows
up once is harmless; it only disables an internal optimisation.
"""

import math
import torch
from torch import nn


# ───────────────────────────────────────────────────────── Positional Encoding ──
class PositionalEncoding(nn.Module):
    """Additive sinusoidal PE (batch-first).  No trainable parameters."""

    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, d]
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # [T, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : [B, T, d_model]

        Returns
        -------
        tensor with same shape, positions added.
        """
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


# ───────────────────────────────────────────────────── Cross-Attention Transformer ──
class CrossAttentionTransformer(nn.Module):
    """
    • Binary variables → encoder (keys/values)
    • Continuous variables → decoder (queries)
    • Predict *only* the next-step continuous vector.
    """

    def __init__(
        self,
        input_size: int = 17,  # total variables
        num_continuous: int = 7,  # continuous variables to predict
        d_model: int = 24,
        nhead: int = 3,
        num_layers: int = 1,  # encoder/decoder depth
        dropout: float = 0.1,
        pred_len: int = 1,  # horizon
        max_len: int = 512,
    ):
        super().__init__()

        self.num_cont = num_continuous
        self.pred_len = pred_len
        self.d_model = d_model

        # ── projections ───────────────────────────────────────────────────
        self.cont_proj = nn.Linear(num_continuous, d_model)  # [B,T,7]→[B,T,d]
        self.bin_proj = nn.Linear(input_size - num_continuous, d_model)

        # ── positional encoding ───────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # ── encoder/decoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        # ── prediction head ───────────────────────────────────────────────
        self.head = nn.Linear(d_model, num_continuous * pred_len)

    # ───────────────────────────────────────────────────────────── forward ──
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : [B, T, 17]  (continuous first, binary afterwards)

        Returns
        -------
        y : [B, pred_len, 7]  next-step continuous prediction
        """
        x_cont = x[..., : self.num_cont]  # [B, T, 7]
        x_bin = x[..., self.num_cont :]  # [B, T, 10]

        # projections + positions
        q = self.pos_enc(self.cont_proj(x_cont))  # queries
        kv = self.pos_enc(self.bin_proj(x_bin))  # keys/values

        # encode binary stream
        memory = self.encoder(kv)

        # decode continuous stream attending to binary memory
        T = q.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(q.device)
        out = self.decoder(
            q, memory, tgt_mask=mask, tgt_is_causal=True
        )  # [B, T, d_model]

        # final prediction from last timestep token
        y_flat = self.head(out[:, -1])  # [B, 7 * pred_len]
        y_flat = y_flat.view(x.size(0), self.pred_len, self.num_cont)
        return y_flat.squeeze(1)  # [B, 7]  (last timestep only)


# ─────────────────────────────────────────────────────────────── smoke test ──
if __name__ == "__main__":
    BATCH, SEQ_LEN = 4, 16
    NUM_CONT, NUM_BIN = 7, 10
    INPUT_SIZE = NUM_CONT + NUM_BIN

    # dummy data
    x_cont = torch.randn(BATCH, SEQ_LEN, NUM_CONT)  # continuous
    x_bin = torch.randint(0, 2, (BATCH, SEQ_LEN, NUM_BIN)).float()  # binary
    x = torch.cat([x_cont, x_bin], dim=-1)  # [B, T, 17]

    model = CrossAttentionTransformer(
        input_size=INPUT_SIZE,
        num_continuous=NUM_CONT,
        d_model=24,
        nhead=3,
        num_layers=1,
        pred_len=1,
    )

    y_hat = model(x)
    print("output shape :", y_hat.shape)  # [4,1,7]

    # check back-prop
    loss = y_hat.mean()
    loss.backward()
    grad_norm = model.head.weight.grad.norm().item()
    print("‖∇‖ on head :", f"{grad_norm:.4f}")
