# Per-(step, variable) vs STEP gated correction -- efficiency summary

Question: does gating the ErrorMLP correction on the VARIABLE axis (correct only the poorly-predicted (step,variable) cells) match/beat the STEP gate's tail (p99) while touching FEWER (step,variable) cells (lower intervention rate), mean non-regressed? Metric = per-step MAE (mean over 10 vars of |pred-true|); tail = p99 over (scenario,step). Intervention rate = fraction of (step,variable) cells corrected (a fired STEP corrects all 10 vars). OP = min p99 s.t. mean <= baseline_mean.

| cell | base p99 | step OP | step p99 | step mean | step interv% | var OP | var p99 | var mean | var interv% | fixed OP | fixed p99 | fixed mean | dp99(var-step) | d_interv%(var-step) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SBO | 0.089529 | b=0.5,q=0.1 | 0.059548 | 0.011756 | 0.26% | b=1.0,q=0.02 | 0.062509 | 0.012669 | 0.10% | NONE | - | - | 0.002961 | -0.16% | per-var gating WORSE p99; comparable efficiency |
| LLOCA_CSP | 0.045492 | b=0.5,q=0.1 | 0.033664 | 0.007294 | 0.29% | b=0.5,q=0.02 | 0.037915 | 0.008350 | 1.11% | K=2,beta=0.5 | 0.032635 | 0.007781 | 0.004251 | 0.82% | per-var gating WORSE p99; LESS efficient (more cells, p99 not better) |
| LLOCA_ECSBS | 0.040949 | b=1.0,q=0.01 | 0.036186 | 0.007118 | 0.01% | b=1.0,q=0.005 | 0.036652 | 0.007117 | 0.09% | NONE | - | - | 0.000466 | 0.07% | per-var gating WORSE p99; LESS efficient (more cells, p99 not better) |
| TLOFW_CSP | 0.055778 | b=1.0,q=0.01 | 0.044887 | 0.012353 | 0.03% | b=0.5,q=0.005 | 0.047020 | 0.012382 | 0.10% | NONE | - | - | 0.002133 | 0.07% | per-var gating WORSE p99; LESS efficient (more cells, p99 not better) |
| TLOFW_ECSBS | 0.056486 | b=1.0,q=0.005 | 0.050728 | 0.011941 | 0.00% | b=0.5,q=0.005 | 0.051260 | 0.011748 | 0.03% | NONE | - | - | 0.000532 | 0.03% | per-var gating WORSE p99; LESS efficient (more cells, p99 not better) |

## Per-cell verdicts
- **SBO**: tail: per-var gating WORSE p99; efficiency: comparable efficiency (step p99=0.059548 ir=0.0026 | var p99=0.062509 ir=0.0010 | fixed p99=-)
- **LLOCA_CSP**: tail: per-var gating WORSE p99; efficiency: LESS efficient (more cells, p99 not better) (step p99=0.033664 ir=0.0029 | var p99=0.037915 ir=0.0111 | fixed p99=0.032635)
- **LLOCA_ECSBS**: tail: per-var gating WORSE p99; efficiency: LESS efficient (more cells, p99 not better) (step p99=0.036186 ir=0.0001 | var p99=0.036652 ir=0.0009 | fixed p99=-)
- **TLOFW_CSP**: tail: per-var gating WORSE p99; efficiency: LESS efficient (more cells, p99 not better) (step p99=0.044887 ir=0.0003 | var p99=0.047020 ir=0.0010 | fixed p99=-)
- **TLOFW_ECSBS**: tail: per-var gating WORSE p99; efficiency: LESS efficient (more cells, p99 not better) (step p99=0.050728 ir=0.0000 | var p99=0.051260 ir=0.0003 | fixed p99=-)

## Cross-cell verdict
- Cells analysed: 5
- Per-var tail vs step: BETTER=0, SAME=0, WORSE=5
- Per-var efficiency vs step: MORE-efficient=0, LESS-efficient=4, comparable=1
- Per-(step,variable) gating tends to WORSEN the tail vs the step gate. This is consistent with the prior that the joint ErrorMLP already self-restricts (outputs ~0 on well-predicted variables), so per-variable gating adds little beyond step gating.
