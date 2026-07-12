# Paper figures

Publication-quality figures for the ErrorMLP / ABC-Transformer accident study.
Regenerate (deterministic, no network, overwrites in place) from `src/`:

```bash
NONINTERACTIVE=1 python experiments/make_paper_figures.py
```

Generator: `src/experiments/make_paper_figures.py`. All numbers are read from the
per-cell analysis JSONs under `<out_root>/<cell>/` (`out_root` + cell list from
`configs/error_mlp_accident.yaml`); nothing is hardcoded, trained, or rolled out.
5 cells: SBO, LLOCA_CSP, LLOCA_ECSBS, TLOFW_CSP, TLOFW_ECSBS. Error unit = per-step
micro-MAE in SCALED space. Palette = Okabe–Ito (colorblind-safe). Each figure is
saved as 300-dpi PNG + vector PDF.

| Figure | What it shows | Source JSON fields |
|---|---|---|
| `fig1_tail_gating` | MAIN result. (a) baseline vs gated-operating-point p99 per cell with the % p99 reduction annotated (−33.5/−26.0/−19.5/−11.6/−10.2%; cells sorted by descending reduction); (b) baseline vs gated mean per cell (mean not regressed, −3 to −19%). Gating cuts the tail while keeping the mean flat. | `tail_analysis.json`: `baseline.{p99,mean}`, `operating_point_strict.{p99,mean,p99_reduction_pct}` |
| `fig2_global_vs_selective` | Why selective, not global. (a) mean micro-MAE vs global β∈{0,.25,.5,.75,1} per cell — monotonic increase, so global β*=0 (β=0 = baseline); (b) SBO at equal β=0.5: global β worsens BOTH mean (+133%) and p99 (+24%), gated (β=0.5,q=0.1) improves BOTH (mean −19%, p99 −33%). | `tail_analysis.json`: `baseline.mean` (β=0), `global_beta["beta=…"].{mean,p99}`, `gated_pred["beta=0.5,q=0.1"].{mean,p99}` |
| `fig3_importance_error_divergence` | Importance ≠ error, 5/5 cells. Per-cell scatter: x = per-input permutation importance (ΔMAE, 10 continuous vars), y = per-output error p99. Spearman ρ(imp,p99) in each title (+0.09/+0.20/−0.30/−0.01/−0.21) — no positive relation. Extreme points labelled (most-important input + 2 hardest outputs). | `variable_analysis.json`: `link_importance_error.importance_continuous`, `output_error.per_out_p99`, `continuous_cols`. ρ computed on the plotted axes (imp vs p99); note `link_importance_error.spearman_import_vs_output_error` is import-vs-MAE (−0.25/+0.32/−0.29/−0.26/−0.27). |
| `fig4_tf_vs_ar` | TF vs AR importance. Per-cell scatter of teacher-forcing ΔMAE (x) vs autoregressive ΔMAE (y) for every input channel (14/15 pts), with y=x reference. Spearman(all channels) in each title (+0.75/+0.31/+0.64/+0.74/+0.86). ZWRB(6) highlighted (drops below y=x under AR), PSGGEN(1) highlighted (rises above). | `ar_permutation_importance.json`: `tf_vs_ar.{feature_cols, tf_dMAE_overall, ar_dMAE_overall, spearman_tf_ar_all_channels}` |
| `fig5_turning_point` | Turning-point hypothesis refuted. (a) H1 p99-tail-share lift at TP, all <1.0 (0.68–0.82) → REFUTED; (b) H2 directional hit-rate overall vs at-TP, TP higher not lower → REFUTED; (c) H3 MISS/HIT downstream cum-err ratio @H=10, only ~1.04–1.15 → weak. | `turning_point_analysis.json`: `pooled.H1.lift_tail_at_tp`, `pooled.H2.{dir_hit_rate_overall,dir_hit_rate_at_tp}`, `pooled.H3["10"].ratio_miss_over_hit` |
