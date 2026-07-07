# ErrorMLP AR-Correction — Results (60min / k=3 / seed=42)

**Idea.** In autoregressive (AR) rollout the ABC-Transformer feeds its own continuous
predictions back into the window, so error compounds. We freeze the trained backbone and
add a small **ErrorMLP** that predicts the per-step error; the correction

```
final_t = backbone_t + beta * ErrorMLP(feats_t)
```

is fed back into the rolling window (and reported), damping the accumulation.

- **Backbone (frozen):** layer8 ABC-Transformer ckpt `epoch=11-...-step=58272.ckpt`
- **ErrorMLP:** in_dim=31 → 2×64 (ReLU, dropout 0.1) → 10, ~6k params (`error_mlp.pt`, 35 KB)
- **Features (31):** backbone_pred(10) + last observed continuous(10) + current binary(10) + `step_idx/300`
- **Target:** per-step error `continuous_y_true − backbone_pred` (open-loop / beta=0 rollout distribution)
- **Loss:** SmoothL1. Trained 30 epochs, held-out val loss 0.0245.

## Headline

| metric | baseline AR | corrected AR (β\*=0.25) | change |
|---|---|---|---|
| micro MAE | 0.06390735 | **0.03440447** | **−46.2 %** |
| macro MAE | 0.06393082 | **0.03444663** | **−46.1 %** |
| micro RMSE | 0.09915485 | 0.05948285 | −40.0 % |
| per-scenario win fraction | — | **838 / 1100 = 76.2 %** | — |

(For reference, teacher-forcing MAE is ≈0.0065; correction closes ~46 % of the TF↔AR gap.)

## Beta sweep (TEST, 1100 scenarios)

| β | micro MAE | macro MAE | micro RMSE |
|---|---|---|---|
| 0.00 (null-op) | 0.06390735 | 0.06393082 | 0.09915485 |
| **0.25 (β\*)** | **0.03440447** | **0.03444663** | **0.05948285** |
| 0.50 | 0.03661589 | 0.03665314 | 0.06569314 |
| 0.75 | 0.04020150 | 0.04023816 | 0.07287661 |
| 1.00 | 0.04255099 | 0.04257908 | 0.07776370 |

The curve is **unimodal with a minimum at β=0.25**: moderate correction damps drift best, while
full correction (β=1.0) over-corrects on the covariate-shifted late-rollout states — exactly the
behaviour the covariate-shift analysis predicted, and evidence the gain is a real mechanism rather
than a lucky grid point. β\* was selected on the disjoint held-out split (1485 train scenarios) and
reported on TEST.

## Checks

- **β=0 null-op (exact):** corrected AR at β=0 reproduces baseline `micro_mae=0.06390735470` /
  `macro_mae=0.06393082624` to 0 error over all 1100 scenarios → the correction wrapper is faithful.
- **Step-index ablation:** zeroing `step_norm` raises β\* micro MAE 0.03440 → 0.04163 (+0.0072, ~+21 %).
  The step feature carries real accumulation signal (result still beats baseline without it).
- **No leakage:** ErrorMLP trained on 8415 train scenarios; β selected on 1485 disjoint held-out
  (train-derived); reported on the separate `_test.csv` (1100 scenarios). Backbone frozen throughout.

## Method / reproduce

```bash
cd src
NONINTERACTIVE=1 python experiments/train_error_mlp.py --interval 60min --k 3 --seed 42
NONINTERACTIVE=1 python experiments/eval_error_mlp.py  --interval 60min --k 3 --seed 42
```

Full machine-readable metrics: `error_mlp_eval_60min_k3_seed42.json` (this dir).

## Honest caveats / next steps

- **Covariate shift (β>0):** the ErrorMLP is trained on the *uncorrected* rollout distribution but
  applied on the *corrected* trajectory. The β sweep's sweet spot (0.25) mitigates this; the exact
  β=0 null-op guarantee holds regardless (it's a property of the wrapper).
- **Optional DAgger Phase-2** (`training.dagger_rounds`, default 0) re-collects errors on the
  corrected trajectory and fine-tunes — the principled fix if we want to push past β=0.25.
- **Scope:** single smoke run (60min/k3/seed42). Next: sweep other intervals (30/15/5min), k∈{10,30},
  and multiple seeds to confirm generality.
