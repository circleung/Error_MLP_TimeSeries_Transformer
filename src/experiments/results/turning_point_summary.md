# Turning-point / directional-error AR failure-mode summary

Frozen backbones, NO retraining, TEST set, reusing the beta=0 (uncorrected) AR rollout. Hypothesis (user's insight): catastrophic AR error blowups happen at **turning points** -- steps where a true trajectory changes direction -- and a **directional miss** at a turning point feeds the wrong value back so the rollout diverges. Per continuous variable: Delta_true[t]=y_true[t]-y_true[t-1], Delta_pred[t]=yhat[t]-y_true[t-1]; a **turning point** is a sign flip of Delta_true (both steps above a per-var deadband eps = p25 of |Delta_true|); a **directional hit** is sign(Delta_pred)==sign(Delta_true). Error unit = per-step abs error, pooled = mean over the 10 vars (== compute_micro_macro per-step MAE); SCALED space. Pooled stats are over (variable, step) turning-point events (NOT 'any-var per step', which saturates). **H1**: do turning points carry disproportionate error (lift). **H2**: is directional accuracy worse at turning points. **H3** (key): does a MISS at a turning point cause a larger DOWNSTREAM error blowup than a HIT.

## Per-cell headline

| cell | scenarios | TP base rate | H1: mean-err TP/non (ratio) | H1: p99-tail share@TP (lift) | H2: dir-hit overall / at-TP (drop) | H3: MISS/HIT downstream cum-err @H=10 (ratio) | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|---|
| SBO | 800/3000 (first-by-id) | 0.114 | 0.0143 / 0.0161 (0.89x) | 0.089 (base 0.114; lift 0.78x) | 0.548 / 0.640 (-0.092) | 0.1863 / 0.1616 (1.15x) | NOT-supported | NOT-supported | SUPPORTED |
| LLOCA_CSP | 800/1500 (first-by-id) | 0.153 | 0.0157 / 0.0058 (2.72x) | 0.114 (base 0.153; lift 0.74x) | 0.550 / 0.589 (-0.039) | 0.1000 / 0.0960 (1.04x) | NOT-supported | NOT-supported | SUPPORTED |
| LLOCA_ECSBS | 800/1500 (first-by-id) | 0.114 | 0.0086 / 0.0066 (1.31x) | 0.094 (base 0.114; lift 0.82x) | 0.610 / 0.652 (-0.042) | 0.0832 / 0.0772 (1.08x) | NOT-supported | NOT-supported | SUPPORTED |
| TLOFW_CSP | 800/1500 (first-by-id) | 0.134 | 0.0173 / 0.0114 (1.51x) | 0.099 (base 0.134; lift 0.74x) | 0.523 / 0.567 (-0.044) | 0.1355 / 0.1284 (1.06x) | NOT-supported | NOT-supported | SUPPORTED |
| TLOFW_ECSBS | 800/1500 (first-by-id) | 0.109 | 0.0115 / 0.0129 (0.89x) | 0.074 (base 0.109; lift 0.68x) | 0.504 / 0.633 (-0.129) | 0.1408 / 0.1306 (1.08x) | NOT-supported | NOT-supported | SUPPORTED |

## SBO

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/3000 scenarios), eps=p25 of |Delta_true| (per var), curvature TP=top-decile |2nd diff|, near=+/-2, tail=p99, rollout approx 25.0s

**H1 (error concentration at turning points):** turning-point base rate = 0.1143 of (var,step) events. Mean per-step error at TP = 0.0143 vs non-TP = 0.0161 (ratio 0.89x; near-TP 0.91x). Turning-point steps carry 0.104 of the total error and 0.089 of the p99 tail (lift 0.78x; near-TP tail share 0.223, lift 0.83x). -> **NOT-supported**

**H2 (directional accuracy at turning points):** dir hit-rate overall = 0.5481 vs at TP = 0.6398 (drop -0.0916; n_dir=4720923, n_dir_TP=719929). -> **NOT-supported**

**H3 (directional MISS at a turning point -> downstream error blowup):**

| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |
|---|---|---|---|---|---|
| 5 | 0.0930 | 0.0806 | 1.15x | 257674 | 457697 |
| 10 | 0.1863 | 0.1616 | 1.15x | 256137 | 454828 |
| 20 | 0.3742 | 0.3249 | 1.15x | 252899 | 449158 |

-> **SUPPORTED** at H=10 (ratio 1.15x)

MISS downstream error trajectory (offset 0..10): 0.0185, 0.0185, 0.0185, 0.0185, 0.0186, 0.0186, 0.0186, 0.0187, 0.0187, 0.0187, 0.0188

HIT  downstream error trajectory (offset 0..10): 0.0159, 0.0160, 0.0161, 0.0161, 0.0161, 0.0161, 0.0162, 0.0162, 0.0162, 0.0163, 0.0163

**Top turning-point-failure variables:**
  - by p99-tail lift at TP: TGRCS(10) (1.651), ZWDC2SG(1) (1.558), TGRCS(15) (1.090)
  - by error ratio TP/non-TP: TGRCS(10) (1.72), PPS (1.08), ZWDC2SG(1) (1.01)
  - by directional hit-rate DROP at TP: TGRCS(15) (0.010), PSGGEN(1) (-0.002), TGRCS(10) (-0.008)
  - by H3 MISS/HIT downstream ratio (H=10): TGRB(17) (1.78), ZWRB(1) (1.39), ZWDC2SG(1) (1.35)

## LLOCA_CSP

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), eps=p25 of |Delta_true| (per var), curvature TP=top-decile |2nd diff|, near=+/-2, tail=p99, rollout approx 26.6s

**H1 (error concentration at turning points):** turning-point base rate = 0.1529 of (var,step) events. Mean per-step error at TP = 0.0157 vs non-TP = 0.0058 (ratio 2.72x; near-TP 2.51x). Turning-point steps carry 0.267 of the total error and 0.114 of the p99 tail (lift 0.74x; near-TP tail share 0.300, lift 0.82x). -> **NOT-supported**

**H2 (directional accuracy at turning points):** dir hit-rate overall = 0.5503 vs at TP = 0.5891 (drop -0.0389; n_dir=4811372, n_dir_TP=980994). -> **NOT-supported**

**H3 (directional MISS at a turning point -> downstream error blowup):**

| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |
|---|---|---|---|---|---|
| 5 | 0.0500 | 0.0479 | 1.04x | 401261 | 574851 |
| 10 | 0.1000 | 0.0960 | 1.04x | 399319 | 571694 |
| 20 | 0.1999 | 0.1919 | 1.04x | 395549 | 565287 |

-> **SUPPORTED** at H=10 (ratio 1.04x)

MISS downstream error trajectory (offset 0..10): 0.0099, 0.0100, 0.0099, 0.0099, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0101, 0.0101

HIT  downstream error trajectory (offset 0..10): 0.0094, 0.0094, 0.0095, 0.0095, 0.0096, 0.0096, 0.0096, 0.0096, 0.0097, 0.0097, 0.0097

**Top turning-point-failure variables:**
  - by p99-tail lift at TP: TGRB(17) (5.793), ZWRB(1) (3.997), ZWRB(6) (1.732)
  - by error ratio TP/non-TP: ZWRB(1) (2.84), TGRB(17) (1.53), TGRCS(15) (1.44)
  - by directional hit-rate DROP at TP: TGRB(17) (0.238), PEX0(17) (0.214), PPS (0.166)
  - by H3 MISS/HIT downstream ratio (H=10): PEX0(17) (1.55), ZWRB(6) (1.53), TGRCS(10) (1.37)

## LLOCA_ECSBS

- controls: SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), eps=p25 of |Delta_true| (per var), curvature TP=top-decile |2nd diff|, near=+/-2, tail=p99, rollout approx 27.2s

**H1 (error concentration at turning points):** turning-point base rate = 0.1144 of (var,step) events. Mean per-step error at TP = 0.0086 vs non-TP = 0.0066 (ratio 1.31x; near-TP 1.39x). Turning-point steps carry 0.136 of the total error and 0.094 of the p99 tail (lift 0.82x; near-TP tail share 0.255, lift 1.00x). -> **NOT-supported**

**H2 (directional accuracy at turning points):** dir hit-rate overall = 0.6099 vs at TP = 0.6520 (drop -0.0420; n_dir=4810917, n_dir_TP=733962). -> **NOT-supported**

**H3 (directional MISS at a turning point -> downstream error blowup):**

| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |
|---|---|---|---|---|---|
| 5 | 0.0415 | 0.0385 | 1.08x | 254242 | 475957 |
| 10 | 0.0832 | 0.0772 | 1.08x | 253066 | 473414 |
| 20 | 0.1671 | 0.1554 | 1.08x | 250661 | 468311 |

-> **SUPPORTED** at H=10 (ratio 1.08x)

MISS downstream error trajectory (offset 0..10): 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0084, 0.0084, 0.0084

HIT  downstream error trajectory (offset 0..10): 0.0075, 0.0076, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0078, 0.0078, 0.0078

**Top turning-point-failure variables:**
  - by p99-tail lift at TP: PPS (14.333), TGRB(17) (5.997), TWSG(1) (2.050)
  - by error ratio TP/non-TP: PPS (2.49), ZWRB(1) (2.16), TGRB(17) (1.87)
  - by directional hit-rate DROP at TP: ZWRB(6) (0.207), TGRB(17) (0.128), TGRCS(15) (0.043)
  - by H3 MISS/HIT downstream ratio (H=10): PPS (1.64), ZWDC2SG(1) (1.50), TGRCS(15) (1.42)

## TLOFW_CSP

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), eps=p25 of |Delta_true| (per var), curvature TP=top-decile |2nd diff|, near=+/-2, tail=p99, rollout approx 16.2s

**H1 (error concentration at turning points):** turning-point base rate = 0.1343 of (var,step) events. Mean per-step error at TP = 0.0173 vs non-TP = 0.0114 (ratio 1.51x; near-TP 1.44x). Turning-point steps carry 0.178 of the total error and 0.099 of the p99 tail (lift 0.74x; near-TP tail share 0.241, lift 0.75x). -> **NOT-supported**

**H2 (directional accuracy at turning points):** dir hit-rate overall = 0.5230 vs at TP = 0.5670 (drop -0.0440; n_dir=4807971, n_dir_TP=861371). -> **NOT-supported**

**H3 (directional MISS at a turning point -> downstream error blowup):**

| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |
|---|---|---|---|---|---|
| 5 | 0.0677 | 0.0641 | 1.06x | 370862 | 485371 |
| 10 | 0.1355 | 0.1284 | 1.06x | 368570 | 482207 |
| 20 | 0.2713 | 0.2573 | 1.05x | 363924 | 475877 |

-> **SUPPORTED** at H=10 (ratio 1.06x)

MISS downstream error trajectory (offset 0..10): 0.0135, 0.0135, 0.0135, 0.0135, 0.0135, 0.0135, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136

HIT  downstream error trajectory (offset 0..10): 0.0127, 0.0127, 0.0128, 0.0128, 0.0128, 0.0128, 0.0129, 0.0129, 0.0129, 0.0129, 0.0130

**Top turning-point-failure variables:**
  - by p99-tail lift at TP: ZWRB(1) (4.103), PPS (1.721), TGRB(17) (1.241)
  - by error ratio TP/non-TP: ZWRB(1) (2.48), PPS (2.31), TGRB(17) (1.26)
  - by directional hit-rate DROP at TP: PEX0(17) (0.085), PPS (0.077), TGRB(17) (0.038)
  - by H3 MISS/HIT downstream ratio (H=10): ZWRB(1) (1.45), ZWRB(6) (1.37), TGRB(17) (1.30)

## TLOFW_ECSBS

- controls: SAMG-01 POSRV, SAMG-02 SG Injection, SAMG-03 RCS Injection, SAMG-06 Spray Pump, SAMG-06 ECSBS
- subsample: first 800 scenario ids ascending (800/1500 scenarios), eps=p25 of |Delta_true| (per var), curvature TP=top-decile |2nd diff|, near=+/-2, tail=p99, rollout approx 17.0s

**H1 (error concentration at turning points):** turning-point base rate = 0.1089 of (var,step) events. Mean per-step error at TP = 0.0115 vs non-TP = 0.0129 (ratio 0.89x; near-TP 0.95x). Turning-point steps carry 0.098 of the total error and 0.074 of the p99 tail (lift 0.68x; near-TP tail share 0.204, lift 0.83x). -> **NOT-supported**

**H2 (directional accuracy at turning points):** dir hit-rate overall = 0.5042 vs at TP = 0.6330 (drop -0.1288; n_dir=4811242, n_dir_TP=698431). -> **NOT-supported**

**H3 (directional MISS at a turning point -> downstream error blowup):**

| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |
|---|---|---|---|---|---|
| 5 | 0.0703 | 0.0651 | 1.08x | 254951 | 439361 |
| 10 | 0.1408 | 0.1306 | 1.08x | 253545 | 436660 |
| 20 | 0.2823 | 0.2619 | 1.08x | 250794 | 431235 |

-> **SUPPORTED** at H=10 (ratio 1.08x)

MISS downstream error trajectory (offset 0..10): 0.0140, 0.0140, 0.0140, 0.0140, 0.0140, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0142

HIT  downstream error trajectory (offset 0..10): 0.0129, 0.0130, 0.0130, 0.0130, 0.0130, 0.0130, 0.0131, 0.0131, 0.0131, 0.0131, 0.0131

**Top turning-point-failure variables:**
  - by p99-tail lift at TP: ZWDC2SG(1) (1.630), TWSG(1) (1.557), TGRCS(10) (1.358)
  - by error ratio TP/non-TP: PPS (1.52), ZWDC2SG(1) (1.27), TGRB(17) (1.21)
  - by directional hit-rate DROP at TP: PSGGEN(1) (0.030), TGRCS(15) (-0.014), TWSG(1) (-0.024)
  - by H3 MISS/HIT downstream ratio (H=10): TGRCS(15) (1.27), PPS (1.23), TGRCS(10) (1.22)

## Cross-cell verdict

- **H1** (errors concentrate at turning points): SUPPORTED in 0/5 cells.
- **H2** (directional accuracy worse at turning points): SUPPORTED in 0/5 cells.
- **H3** (a directional MISS at a turning point -> larger downstream error blowup than a HIT): SUPPORTED in 5/5 cells.

Per-cell H3 ratio (MISS/HIT downstream cum-err @H=10):
  - SBO: 1.15x (SUPPORTED)
  - LLOCA_CSP: 1.04x (SUPPORTED)
  - LLOCA_ECSBS: 1.08x (SUPPORTED)
  - TLOFW_CSP: 1.06x (SUPPORTED)
  - TLOFW_ECSBS: 1.08x (SUPPORTED)

Cross-cell recurrence of top turning-point-failure variables:
**Most-recurrent by H3 MISS/HIT ratio (top-3):**
  - TGRB(17): 2/5
  - ZWRB(1): 2/5
  - ZWDC2SG(1): 2/5
  - ZWRB(6): 2/5
  - TGRCS(10): 2/5
  - PPS: 2/5
  - TGRCS(15): 2/5
  - PEX0(17): 1/5

**Most-recurrent by directional hit-rate DROP at TP (top-3):**
  - TGRCS(15): 3/5
  - TGRB(17): 3/5
  - PSGGEN(1): 2/5
  - PEX0(17): 2/5
  - PPS: 2/5
  - TGRCS(10): 1/5
  - ZWRB(6): 1/5
  - TWSG(1): 1/5

**Overall:** H1 REFUTED (0/5); H2 REFUTED (0/5); H3 SUPPORTED (5/5). The user's hypothesis is that directional misses at turning points cause the blowups; it is best judged by H3 (does a MISS blow up downstream more than a HIT) together with H1 (are turning points where the error lives).

