# Per-accident-type variable-importance + error-attribution summary

Frozen backbones, NO retraining, TEST set. **Importance** = permutation importance in teacher-forcing next-step (dMAE = increase in next-step MAE when an INPUT channel is shuffled across test windows; avg of 3 shuffles). **Output error** = per-output MAE/p99 of the baseline uncorrected AR rollout (beta=0). **Worst-scenario** = top-20 scenarios by per-scenario mean AR error; share = fraction of the pooled worst-scenario error carried by each output variable. **Link** = Spearman rank correlation over the 10 continuous vars (which are both inputs and outputs) between input importance and per-output AR error / worst-scenario share.

## Per-cell headline

| cell | top-3 important inputs (dMAE) | top-3 hardest outputs (MAE) | worst-20 top contributor | top-3 cum share | import<->output rho | verdict-short |
|---|---|---|---|---|---|---|
| SBO | PSGGEN(1) (0.044101), ZWRB(6) (0.043797), SAMG-02 SG Injection (0.038727) | TGRCS(15) (0.031372), TGRCS(10) (0.025783), PEX0(17) (0.024021) | PEX0(17) (0.192) | 0.552 | -0.248 | DIVERGENT |
| LLOCA_CSP | SAMG-02 SG Injection (0.039048), ZWRB(6) (0.026526), TGRB(17) (0.025287) | TGRCS(15) (0.043540), TGRCS(10) (0.020494), ZWRB(6) (0.009151) | TGRCS(15) (0.661) | 0.921 | 0.321 | DIVERGENT |
| LLOCA_ECSBS | ZWRB(6) (0.041661), SAMG-02 SG Injection (0.025127), TWSG(1) (0.022023) | TGRCS(15) (0.021816), TGRCS(10) (0.012079), TGRB(17) (0.008264) | TGRCS(15) (0.347) | 0.608 | -0.285 | DIVERGENT |
| TLOFW_CSP | TWSG(1) (0.031106), PSGGEN(1) (0.024971), ZWRB(6) (0.022822) | TGRCS(10) (0.046641), TGRCS(15) (0.039796), ZWRB(6) (0.008787) | PEX0(17) (0.368) | 0.584 | -0.261 | DIVERGENT |
| TLOFW_ECSBS | ZWRB(6) (0.038383), TWSG(1) (0.028059), PSGGEN(1) (0.023946) | TGRCS(15) (0.030949), TGRCS(10) (0.028153), PEX0(17) (0.018589) | PEX0(17) (0.474) | 0.742 | -0.273 | DIVERGENT |

## SBO

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 ECSBS  (n_scenarios=3000, n_test_windows=2361000)
- baseline teacher-forcing MAE: 0.001063

**Top-3 most important INPUT variables (permutation dMAE):**
  - #1 PSGGEN(1): dMAE=0.044101
  - #2 ZWRB(6): dMAE=0.043797
  - #3 SAMG-02 SG Injection: dMAE=0.038727

**Top-3 hardest OUTPUT variables (AR rollout):**
  - #1 TGRCS(15): MAE=0.031372, p99=0.181289
  - #2 TGRCS(10): MAE=0.025783, p99=0.283134
  - #3 PEX0(17): MAE=0.024021, p99=0.205806

**Hardest OUTPUT variables by p99 (tail):**
  - #1 TGRCS(10): p99=0.283134
  - #2 ZWRB(6): p99=0.235431
  - #3 PEX0(17): p99=0.205806

**Worst-20 scenarios error attribution** (avg per-scenario mean err=0.073485):
  - PEX0(17): share=0.1924
  - ZWRB(6): share=0.1863
  - TGRCS(10): share=0.1729
  - top-3 cumulative share: 0.5516

**Link importance <-> error:**
  - most important input: PSGGEN(1)
  - hardest output: TGRCS(15) (the important input is #9 hardest output)
  - top worst-scenario driver: PEX0(17) (the important input carries worst-share rank #8)
  - Spearman(import, output-error) = -0.248 (p=0.489)
  - Spearman(import, worst-share) = 0.006 (p=0.987)
  - VERDICT: DIVERGENT: most important input = PSGGEN(1); hardest output = TGRCS(15) (import var is #9 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #8).

## LLOCA_CSP

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS  (n_scenarios=1500, n_test_windows=1203000)
- baseline teacher-forcing MAE: 0.001336

**Top-3 most important INPUT variables (permutation dMAE):**
  - #1 SAMG-02 SG Injection: dMAE=0.039048
  - #2 ZWRB(6): dMAE=0.026526
  - #3 TGRB(17): dMAE=0.025287

**Top-3 hardest OUTPUT variables (AR rollout):**
  - #1 TGRCS(15): MAE=0.043540, p99=0.294961
  - #2 TGRCS(10): MAE=0.020494, p99=0.165546
  - #3 ZWRB(6): MAE=0.009151, p99=0.053469

**Hardest OUTPUT variables by p99 (tail):**
  - #1 TGRCS(15): p99=0.294961
  - #2 TGRCS(10): p99=0.165546
  - #3 ZWRB(6): p99=0.053469

**Worst-20 scenarios error attribution** (avg per-scenario mean err=0.030213):
  - TGRCS(15): share=0.6612
  - TGRCS(10): share=0.2131
  - ZWRB(6): share=0.0468
  - top-3 cumulative share: 0.9211

**Link importance <-> error:**
  - most important input: ZWRB(6)
  - hardest output: TGRCS(15) (the important input is #3 hardest output)
  - top worst-scenario driver: TGRCS(15) (the important input carries worst-share rank #3)
  - Spearman(import, output-error) = 0.321 (p=0.365)
  - Spearman(import, worst-share) = 0.321 (p=0.365)
  - VERDICT: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #3 hardest); top worst-scenario driver = TGRCS(15) (import var carries share-rank #3).

## LLOCA_ECSBS

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS  (n_scenarios=1500, n_test_windows=1203000)
- baseline teacher-forcing MAE: 0.001318

**Top-3 most important INPUT variables (permutation dMAE):**
  - #1 ZWRB(6): dMAE=0.041661
  - #2 SAMG-02 SG Injection: dMAE=0.025127
  - #3 TWSG(1): dMAE=0.022023

**Top-3 hardest OUTPUT variables (AR rollout):**
  - #1 TGRCS(15): MAE=0.021816, p99=0.215082
  - #2 TGRCS(10): MAE=0.012079, p99=0.131482
  - #3 TGRB(17): MAE=0.008264, p99=0.056543

**Hardest OUTPUT variables by p99 (tail):**
  - #1 TGRCS(15): p99=0.215082
  - #2 TGRCS(10): p99=0.131482
  - #3 TGRB(17): p99=0.056543

**Worst-20 scenarios error attribution** (avg per-scenario mean err=0.044926):
  - TGRCS(15): share=0.3473
  - TGRCS(10): share=0.1458
  - ZWRB(6): share=0.1145
  - top-3 cumulative share: 0.6076

**Link importance <-> error:**
  - most important input: ZWRB(6)
  - hardest output: TGRCS(15) (the important input is #8 hardest output)
  - top worst-scenario driver: TGRCS(15) (the important input carries worst-share rank #3)
  - Spearman(import, output-error) = -0.285 (p=0.425)
  - Spearman(import, worst-share) = 0.333 (p=0.347)
  - VERDICT: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #8 hardest); top worst-scenario driver = TGRCS(15) (import var carries share-rank #3).

## TLOFW_CSP

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS  (n_scenarios=1500, n_test_windows=1202596)
- baseline teacher-forcing MAE: 0.000873

**Top-3 most important INPUT variables (permutation dMAE):**
  - #1 TWSG(1): dMAE=0.031106
  - #2 PSGGEN(1): dMAE=0.024971
  - #3 ZWRB(6): dMAE=0.022822

**Top-3 hardest OUTPUT variables (AR rollout):**
  - #1 TGRCS(10): MAE=0.046641, p99=0.198827
  - #2 TGRCS(15): MAE=0.039796, p99=0.187703
  - #3 ZWRB(6): MAE=0.008787, p99=0.054286

**Hardest OUTPUT variables by p99 (tail):**
  - #1 TGRCS(10): p99=0.198827
  - #2 TGRCS(15): p99=0.187703
  - #3 PSGGEN(1): p99=0.152795

**Worst-20 scenarios error attribution** (avg per-scenario mean err=0.075173):
  - PEX0(17): share=0.3677
  - TGRB(17): share=0.1164
  - TGRCS(10): share=0.0999
  - top-3 cumulative share: 0.5840

**Link importance <-> error:**
  - most important input: TWSG(1)
  - hardest output: TGRCS(10) (the important input is #9 hardest output)
  - top worst-scenario driver: PEX0(17) (the important input carries worst-share rank #4)
  - Spearman(import, output-error) = -0.261 (p=0.467)
  - Spearman(import, worst-share) = 0.139 (p=0.701)
  - VERDICT: DIVERGENT: most important input = TWSG(1); hardest output = TGRCS(10) (import var is #9 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #4).

## TLOFW_ECSBS

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS  (n_scenarios=1500, n_test_windows=1203000)
- baseline teacher-forcing MAE: 0.001055

**Top-3 most important INPUT variables (permutation dMAE):**
  - #1 ZWRB(6): dMAE=0.038383
  - #2 TWSG(1): dMAE=0.028059
  - #3 PSGGEN(1): dMAE=0.023946

**Top-3 hardest OUTPUT variables (AR rollout):**
  - #1 TGRCS(15): MAE=0.030949, p99=0.173969
  - #2 TGRCS(10): MAE=0.028153, p99=0.154272
  - #3 PEX0(17): MAE=0.018589, p99=0.114556

**Hardest OUTPUT variables by p99 (tail):**
  - #1 TGRCS(15): p99=0.173969
  - #2 TGRCS(10): p99=0.154272
  - #3 TGRB(17): p99=0.120344

**Worst-20 scenarios error attribution** (avg per-scenario mean err=0.070478):
  - PEX0(17): share=0.4741
  - TGRB(17): share=0.1720
  - TGRCS(15): share=0.0958
  - top-3 cumulative share: 0.7419

**Link importance <-> error:**
  - most important input: ZWRB(6)
  - hardest output: TGRCS(15) (the important input is #6 hardest output)
  - top worst-scenario driver: PEX0(17) (the important input carries worst-share rank #8)
  - Spearman(import, output-error) = -0.273 (p=0.446)
  - Spearman(import, worst-share) = -0.297 (p=0.405)
  - VERDICT: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #6 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #8).

## Cross-cell recurrence (how many of the 5 cells put each variable in its top-3)

**Recurrently IMPORTANT inputs (top-3):**
  - ZWRB(6): 5/5
  - PSGGEN(1): 3/5
  - SAMG-02 SG Injection: 3/5
  - TWSG(1): 3/5
  - TGRB(17): 1/5

**Recurrently HARDEST outputs (top-3 MAE):**
  - TGRCS(15): 5/5
  - TGRCS(10): 5/5
  - PEX0(17): 2/5
  - ZWRB(6): 2/5
  - TGRB(17): 1/5

**Recurrent worst-scenario DRIVERS (top-3 share):**
  - TGRCS(10): 4/5
  - PEX0(17): 3/5
  - ZWRB(6): 3/5
  - TGRCS(15): 3/5
  - TGRB(17): 2/5

## Overall link verdict: 0/5 cells fully ALIGNED (most important input == hardest output == top worst-scenario driver)
- **SBO**: DIVERGENT: most important input = PSGGEN(1); hardest output = TGRCS(15) (import var is #9 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #8).
- **LLOCA_CSP**: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #3 hardest); top worst-scenario driver = TGRCS(15) (import var carries share-rank #3).
- **LLOCA_ECSBS**: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #8 hardest); top worst-scenario driver = TGRCS(15) (import var carries share-rank #3).
- **TLOFW_CSP**: DIVERGENT: most important input = TWSG(1); hardest output = TGRCS(10) (import var is #9 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #4).
- **TLOFW_ECSBS**: DIVERGENT: most important input = ZWRB(6); hardest output = TGRCS(15) (import var is #6 hardest); top worst-scenario driver = PEX0(17) (import var carries share-rank #8).

