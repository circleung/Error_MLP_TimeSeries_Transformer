# AR-rollout vs teacher-forcing permutation-importance summary

Frozen backbones, NO retraining, TEST set. **AR importance** = permutation importance in AR ROLLOUT: an INPUT channel is shuffled across the rollout scenarios and *held shuffled through the whole lockstep rollout*, so the corruption propagates through the model's own feedback (continuous channels are the fed-back predictions; controls are the injected known-future). dMAE_AR = increase in the rollout MAE (mean over scenario x step x 10 outputs of |pred-true|) when a channel is corrupted, averaged over the fixed shuffle seeds. **TF importance** (committed, variable_analysis.json) = the same permutation but a SINGLE teacher-forcing forward (no feedback). The question: does feedback + error compounding change WHICH variables matter, and does the TF importance<->output-error DIVERGENCE survive under AR importance?

## Per-cell headline

| cell | subsample | baseline AR MAE | top-3 AR important | top-3 TF important | Spearman(TF,AR) all | Spearman(TF,AR) cont | top-3 overlap | AR import<->output rho | TF import<->output rho | divergence under AR? |
|---|---|---|---|---|---|---|---|---|---|---|
| SBO | 800/3000 (first-by-id) | 0.015753 | PSGGEN(1) (0.087495), TWSG(1) (0.084119), TGRCS(10) (0.069821) | PSGGEN(1), ZWRB(6), SAMG-02 SG Injection | 0.754 | 0.661 | 1/3 ['PSGGEN(1)'] | -0.055 | -0.248 | YES |
| LLOCA_CSP | 800/1500 (first-by-id) | 0.008995 | PSGGEN(1) (0.056960), SAMG-02 SG Injection (0.040363), TGRB(17) (0.034034) | SAMG-02 SG Injection, ZWRB(6), TGRB(17) | 0.310 | -0.055 | 2/3 ['SAMG-02 SG Injection', 'TGRB(17)'] | -0.733 | 0.321 | YES |
| LLOCA_ECSBS | 800/1500 (first-by-id) | 0.007221 | PSGGEN(1) (0.067019), TWSG(1) (0.062510), SAMG-02 SG Injection (0.039621) | ZWRB(6), SAMG-02 SG Injection, TWSG(1) | 0.644 | 0.442 | 2/3 ['SAMG-02 SG Injection', 'TWSG(1)'] | -0.709 | -0.285 | YES |
| TLOFW_CSP | 800/1500 (first-by-id) | 0.013033 | PSGGEN(1) (0.129910), TWSG(1) (0.089002), ZWDC2SG(1) (0.069553) | TWSG(1), PSGGEN(1), ZWRB(6) | 0.736 | 0.661 | 2/3 ['PSGGEN(1)', 'TWSG(1)'] | -0.576 | -0.261 | YES |
| TLOFW_ECSBS | 800/1500 (first-by-id) | 0.012741 | ZWDC2SG(1) (0.084792), TWSG(1) (0.084055), PSGGEN(1) (0.075861) | ZWRB(6), TWSG(1), PSGGEN(1) | 0.857 | 0.806 | 2/3 ['PSGGEN(1)', 'TWSG(1)'] | -0.442 | -0.273 | YES |

## SBO

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/3000 scenarios), maxL=787, n_shuffles=2 seeds=[1234, 5678], approx GPU=339.7s
- baseline AR rollout MAE (subsample): 0.015753

**Top-3 AR-rollout important INPUT variables (dMAE_AR):**
  - #1 PSGGEN(1): dMAE_AR=0.087495
  - #2 TWSG(1): dMAE_AR=0.084119
  - #3 TGRCS(10): dMAE_AR=0.069821

**Top-3 teacher-forcing important INPUT variables (committed dMAE):**
  - #1 PSGGEN(1)
  - #2 ZWRB(6)
  - #3 SAMG-02 SG Injection

**Spearman(TF, AR)** over ALL 14 input channels = 0.754 (p=0.002); over the 10 continuous only = 0.661 (p=0.038)
- top-3 overlap: 1/3 ['PSGGEN(1)']

**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**
  - ZWRB(6) (continuous): TF #2 -> AR #9 (shift -7)
  - TGRCS(10) (continuous): TF #6 -> AR #3 (shift +3)
  - TGRCS(15) (continuous): TF #9 -> AR #6 (shift +3)
  - ZWRB(1) (continuous): TF #7 -> AR #10 (shift -3)
  - PEX0(17) (continuous): TF #8 -> AR #11 (shift -3)

**Importance <-> output-error link under AR:**
  - most important AR input: PSGGEN(1) (it is the #9 hardest output)
  - hardest output: TGRCS(15)
  - Spearman(AR-import, output-error) = -0.055 (p=0.881)
  - committed Spearman(TF-import, output-error) = -0.248 (TF verdict all_aligned=False)
  - VERDICT: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #9 hardest output).

## LLOCA_CSP

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), maxL=802, n_shuffles=2 seeds=[1234, 5678], approx GPU=579.8s
- baseline AR rollout MAE (subsample): 0.008995

**Top-3 AR-rollout important INPUT variables (dMAE_AR):**
  - #1 PSGGEN(1): dMAE_AR=0.056960
  - #2 SAMG-02 SG Injection: dMAE_AR=0.040363
  - #3 TGRB(17): dMAE_AR=0.034034

**Top-3 teacher-forcing important INPUT variables (committed dMAE):**
  - #1 SAMG-02 SG Injection
  - #2 ZWRB(6)
  - #3 TGRB(17)

**Spearman(TF, AR)** over ALL 14 input channels = 0.310 (p=0.281); over the 10 continuous only = -0.055 (p=0.881)
- top-3 overlap: 2/3 ['SAMG-02 SG Injection', 'TGRB(17)']

**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**
  - ZWRB(6) (continuous): TF #2 -> AR #12 (shift -10)
  - SAMG-03 RCS Injection (control): TF #13 -> AR #5 (shift +8)
  - PSGGEN(1) (continuous): TF #7 -> AR #1 (shift +6)
  - ZWRB(1) (continuous): TF #4 -> AR #10 (shift -6)
  - PPS (continuous): TF #12 -> AR #7 (shift +5)

**Importance <-> output-error link under AR:**
  - most important AR input: PSGGEN(1) (it is the #10 hardest output)
  - hardest output: TGRCS(15)
  - Spearman(AR-import, output-error) = -0.733 (p=0.016)
  - committed Spearman(TF-import, output-error) = 0.321 (TF verdict all_aligned=False)
  - VERDICT: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #10 hardest output).

## LLOCA_ECSBS

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), maxL=802, n_shuffles=2 seeds=[1234, 5678], approx GPU=1120.3s
- baseline AR rollout MAE (subsample): 0.007221

**Top-3 AR-rollout important INPUT variables (dMAE_AR):**
  - #1 PSGGEN(1): dMAE_AR=0.067019
  - #2 TWSG(1): dMAE_AR=0.062510
  - #3 SAMG-02 SG Injection: dMAE_AR=0.039621

**Top-3 teacher-forcing important INPUT variables (committed dMAE):**
  - #1 ZWRB(6)
  - #2 SAMG-02 SG Injection
  - #3 TWSG(1)

**Spearman(TF, AR)** over ALL 14 input channels = 0.644 (p=0.013); over the 10 continuous only = 0.442 (p=0.200)
- top-3 overlap: 2/3 ['SAMG-02 SG Injection', 'TWSG(1)']

**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**
  - TGRCS(10) (continuous): TF #4 -> AR #10 (shift -6)
  - TGRCS(15) (continuous): TF #7 -> AR #13 (shift -6)
  - ZWDC2SG(1) (continuous): TF #11 -> AR #5 (shift +6)
  - SAMG-06 ECSBS (control): TF #12 -> AR #7 (shift +5)
  - PSGGEN(1) (continuous): TF #5 -> AR #1 (shift +4)

**Importance <-> output-error link under AR:**
  - most important AR input: PSGGEN(1) (it is the #10 hardest output)
  - hardest output: TGRCS(15)
  - Spearman(AR-import, output-error) = -0.709 (p=0.022)
  - committed Spearman(TF-import, output-error) = -0.285 (TF verdict all_aligned=False)
  - VERDICT: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #10 hardest output).

## TLOFW_CSP

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), maxL=802, n_shuffles=2 seeds=[1234, 5678], approx GPU=943.4s
- baseline AR rollout MAE (subsample): 0.013033

**Top-3 AR-rollout important INPUT variables (dMAE_AR):**
  - #1 PSGGEN(1): dMAE_AR=0.129910
  - #2 TWSG(1): dMAE_AR=0.089002
  - #3 ZWDC2SG(1): dMAE_AR=0.069553

**Top-3 teacher-forcing important INPUT variables (committed dMAE):**
  - #1 TWSG(1)
  - #2 PSGGEN(1)
  - #3 ZWRB(6)

**Spearman(TF, AR)** over ALL 15 input channels = 0.736 (p=0.002); over the 10 continuous only = 0.661 (p=0.038)
- top-3 overlap: 2/3 ['PSGGEN(1)', 'TWSG(1)']

**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**
  - ZWRB(6) (continuous): TF #3 -> AR #12 (shift -9)
  - SAMG-06 Spray Pump (control): TF #11 -> AR #6 (shift +5)
  - TGRCS(15) (continuous): TF #10 -> AR #13 (shift -3)
  - SAMG-03 RCS Injection (control): TF #12 -> AR #9 (shift +3)
  - PPS (continuous): TF #13 -> AR #11 (shift +2)

**Importance <-> output-error link under AR:**
  - most important AR input: PSGGEN(1) (it is the #8 hardest output)
  - hardest output: TGRCS(10)
  - Spearman(AR-import, output-error) = -0.576 (p=0.082)
  - committed Spearman(TF-import, output-error) = -0.261 (TF verdict all_aligned=False)
  - VERDICT: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(10) (AR-import var is #8 hardest output).

## TLOFW_ECSBS

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), maxL=802, n_shuffles=2 seeds=[1234, 5678], approx GPU=978.8s
- baseline AR rollout MAE (subsample): 0.012741

**Top-3 AR-rollout important INPUT variables (dMAE_AR):**
  - #1 ZWDC2SG(1): dMAE_AR=0.084792
  - #2 TWSG(1): dMAE_AR=0.084055
  - #3 PSGGEN(1): dMAE_AR=0.075861

**Top-3 teacher-forcing important INPUT variables (committed dMAE):**
  - #1 ZWRB(6)
  - #2 TWSG(1)
  - #3 PSGGEN(1)

**Spearman(TF, AR)** over ALL 15 input channels = 0.857 (p=0.000); over the 10 continuous only = 0.806 (p=0.005)
- top-3 overlap: 2/3 ['PSGGEN(1)', 'TWSG(1)']

**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**
  - PEX0(17) (continuous): TF #6 -> AR #11 (shift -5)
  - SAMG-06 ECSBS (control): TF #11 -> AR #7 (shift +4)
  - TGRCS(15) (continuous): TF #10 -> AR #13 (shift -3)
  - ZWDC2SG(1) (continuous): TF #4 -> AR #1 (shift +3)
  - ZWRB(6) (continuous): TF #1 -> AR #4 (shift -3)

**Importance <-> output-error link under AR:**
  - most important AR input: ZWDC2SG(1) (it is the #7 hardest output)
  - hardest output: TGRCS(15)
  - Spearman(AR-import, output-error) = -0.442 (p=0.200)
  - committed Spearman(TF-import, output-error) = -0.273 (TF verdict all_aligned=False)
  - VERDICT: DIVERGENT-under-AR: most important AR input = ZWDC2SG(1) but hardest output = TGRCS(15) (AR-import var is #7 hardest output).

## Cross-cell recurrence (how many of the 5 cells put each variable in its top-3)

**Recurrently AR-important inputs (top-3):**
  - PSGGEN(1): 5/5
  - TWSG(1): 4/5
  - SAMG-02 SG Injection: 2/5
  - ZWDC2SG(1): 2/5
  - TGRCS(10): 1/5
  - TGRB(17): 1/5

**Recurrently TF-important inputs (top-3):**
  - ZWRB(6): 5/5
  - PSGGEN(1): 3/5
  - SAMG-02 SG Injection: 3/5
  - TWSG(1): 3/5
  - TGRB(17): 1/5

## Overall verdict

- mean Spearman(TF, AR) over all input channels = 0.660 (per cell: 0.754, 0.310, 0.644, 0.736, 0.857)
- mean Spearman(TF, AR) over the 10 continuous = 0.503 (per cell: 0.661, -0.055, 0.442, 0.661, 0.806)
- importance<->output-error DIVERGENCE persists under AR in 5/5 cells (most important AR input != hardest output)

- **SBO**: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #9 hardest output).
- **LLOCA_CSP**: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #10 hardest output).
- **LLOCA_ECSBS**: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(15) (AR-import var is #10 hardest output).
- **TLOFW_CSP**: DIVERGENT-under-AR: most important AR input = PSGGEN(1) but hardest output = TGRCS(10) (AR-import var is #8 hardest output).
- **TLOFW_ECSBS**: DIVERGENT-under-AR: most important AR input = ZWDC2SG(1) but hardest output = TGRCS(15) (AR-import var is #7 hardest output).

