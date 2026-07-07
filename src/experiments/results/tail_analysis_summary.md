# Per-accident-type GATED tail-correction summary

Goal: cut the TAIL (p99/max of the per-step error dist) on TEST via a SELECTIVE (gated) ErrorMLP correction, keeping the mean (micro_mae) non-regressed. baseline = uncorrected AR. op = best PREDICTED-gate operating point s.t. mean <= baseline_mean (strict). oracle = gate-on-true-error p99 ceiling (detector ceiling).

| cell | baseline_mean | baseline_p99 | baseline_max | op(beta*,q*) | op_mean | op_p99 | op_max | op %p99-red | cuts_tail | oracle_p99 | oracle %p99-red |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SBO | 0.014547 | 0.089529 | 0.291791 | beta=0.5,q=0.1 | 0.011756 | 0.059548 | 0.206028 | 33.49 | True | 0.070443 | 21.32 |
| LLOCA_CSP | 0.008987 | 0.045492 | 0.097411 | beta=0.5,q=0.1 | 0.007294 | 0.033664 | 0.096069 | 26.00 | True | 0.030676 | 32.57 |
| LLOCA_ECSBS | 0.007404 | 0.040949 | 0.265818 | beta=1.0,q=0.01 | 0.007118 | 0.036186 | 0.220257 | 11.63 | True | 0.038940 | 4.91 |
| TLOFW_CSP | 0.012726 | 0.055778 | 0.232420 | beta=1.0,q=0.01 | 0.012353 | 0.044887 | 0.232420 | 19.53 | True | 0.051375 | 7.89 |
| TLOFW_ECSBS | 0.012306 | 0.056486 | 0.255377 | beta=1.0,q=0.005 | 0.011941 | 0.050728 | 0.233414 | 10.19 | True | 0.049675 | 12.06 |

## Per-cell verdict
- **SBO**: gating cuts the tail at mean-neutral operating point
- **LLOCA_CSP**: gating cuts the tail at mean-neutral operating point
- **LLOCA_ECSBS**: gating cuts the tail at mean-neutral operating point
- **TLOFW_CSP**: gating cuts the tail at mean-neutral operating point
- **TLOFW_ECSBS**: gating cuts the tail at mean-neutral operating point
