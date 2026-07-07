import torch
import torch.nn as nn


class ErrorMLP(nn.Module):
    """Learns backbone next-step continuous error under AR rollout drift.

    Input  = concat(backbone_pred[10], last_obs_cont[10], cur_binary[10], step_norm[1]) = 31
             step_norm = absolute_rollout_step / STEP_NORM_CONST (fixed 300.0; see R6).
    Output = predicted error[10] (continuous_y_true - backbone_pred)
    """

    def __init__(self, in_dim=31, hidden_dim=64, out_dim=10,
                 num_hidden=2, dropout=0.1, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers, d = [], in_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(d, hidden_dim), act(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats):        # feats: [B, in_dim]
        return self.net(feats)       # [B, out_dim]
