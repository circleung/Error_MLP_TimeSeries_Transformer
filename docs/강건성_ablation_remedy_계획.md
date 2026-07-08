# 극단 시나리오 강건화 — 변수 ablation 인과검증 & 꼬리 보정 계획서

> 대상 레포: `Error_MLP_TimeSeries_Transformer` (main)
> 실행 환경: conda `transformer_env` (PyTorch 2.5.1+cu121, PL 2.5.3), A100-40GB 단일, `NONINTERACTIVE=1`, `src/`에서 실행
> 상태: **ralplan 합의 승인 완료(Planner→Architect→Critic 3회 반복)** · 비싼 재학습 실행은 사용자 승인 대기
> 상세 원본(영문): `.omc/plans/robustness_ablation_remedy.md`

---

## 0. 한 줄 요약

사고유형별 ABC-Transformer(seq50) 백본은 **평균 정확도는 이미 최고 수준**이지만 **드물게 크게 틀리는 극단(꼬리) 순간**이 존재한다. 이 꼬리를 **평균은 유지하면서** 줄이는 것이 목표다. 검증 결과 **전역 보정은 무효, "큰 오차가 예측되는 소수 스텝에만 거는 게이팅 보정"이 5셀 전부에서 꼬리(p99)를 −10~34% 줄이며 평균도 개선**했다. 이제 (Phase R) 이 게이팅 보정을 병목 셀에 강화하고, (Phase A) 변수 중요도가 실제 인과 효과인지 재학습 ablation으로 검증한다.

---

## 1. 지금까지의 발견 (근거)

### 1.1 ErrorMLP 보정 아이디어
백본은 다음 스텝 연속변수 10개를 예측한다. 자기회귀(AR) rollout에서 자기 예측을 되먹이면 오차가 누적된다. 아이디어: 백본을 고정(freeze)하고 작은 **ErrorMLP**를 붙여
`최종예측 = 백본예측 + β · ErrorMLP(feats)` 로 보정하고, 보정값을 window에 되먹인다.

### 1.2 두 세팅의 상반된 결과
| 세팅 | 백본 AR 오차 | 전역 β 보정 | 게이팅(선택적) 보정 |
|---|---|---|---|
| 60min 통합(약한 백본) | micro 0.064 | **−46%** (전역이 통함) | — |
| **사고유형별 seq50(강한 백본)** | micro 0.0074~0.0146 | **무효 (β\*=0)** | **꼬리 p99 −10~34%, 평균도 개선** |

강한 백본은 평균 오차가 이미 작아, 전역 보정은 쉬운 다수 스텝에 노이즈만 더해 무효다. 그러나 **예측오차 상위 q% 스텝에만 보정을 거는 게이팅**은 5셀 전부에서 꼬리를 줄이며 평균을 유지/개선했다.

### 1.3 게이팅 보정 실측 (사고유형별, mean-neutral 운영점)
| 셀 | baseline p99 | 게이팅 p99 | **p99 감소** | 운영점(β, top-q) |
|---|---|---|---|---|
| SBO | 0.0895 | 0.0595 | **−33.5%** | β0.5, q0.10 |
| LLOCA_CSP | 0.0455 | 0.0337 | **−26.0%** | β0.5, q0.10 |
| TLOFW_CSP | 0.0558 | 0.0449 | **−19.5%** | β1.0, q0.01 |
| LLOCA_ECSBS | 0.0409 | 0.0362 | −11.6% | β1.0, q0.01 |
| TLOFW_ECSBS | 0.0565 | 0.0507 | −10.2% | β1.0, q0.005 |

Oracle-gap 진단: **SBO·LLOCA_CSP**는 검출기·보정기 모두 양호. **TLOFW_CSP·TLOFW_ECSBS·LLOCA_ECSBS**는 보정기 자체가 병목 → tail-loss/DAgger가 유효한 셀.

### 1.4 변수 중요도 ↔ 오차 = 분리 (5/5 셀)
| 축 | 지배 변수 |
|---|---|
| 예측에 중요한 입력(permutation ΔMAE) | **ZWRB(6)[5/5]**, PSGGEN(1), SAMG-02 SG주입, TWSG(1) — 수위·압력·제어 |
| 가장 못 맞추는 출력(MAE/p99) | **핵심 온도 TGRCS(15)·TGRCS(10)[5/5]** |
| 최악 시나리오 오차 주범 | TGRCS(10)[4/5], **PEX0(17)**, ZWRB(6) |

Spearman(중요도, 출력오차) ≈ 0/음수. **모델이 의존하는 입력변수와, 못 맞추는 출력·극단 주범 변수는 서로 다르다.** → 단일 변수 개입으로 둘 다 잡을 수 없음. **강건성은 보정기(remedy)가, 인과 검증은 ablation이 담당**하도록 분리 설계.

---

## 2. 목표

1. **인과 검증**: 변수 중요도(permutation)가 실제 인과 효과인지 "변수 제거 후 백본 재학습" ablation으로 확인.
2. **강건성 향상**: 극단/꼬리 시나리오의 큰 이탈을 평균 유지하며 감소.
3. **극단 시나리오 보완법 설계**: 위 근거로 실효적 remedy 확정.

---

## 3. 접근 — 2단계 (remedy 먼저, ablation은 게이트)

### Phase R — Remedy (강건성의 실제 레버, ~2 GPU-h, **먼저 실행**)
- **베이스라인**: 검증된 게이팅 ErrorMLP(§1.3)를 셀별 운영점으로 확정.
- **병목 3셀(TLOFW_CSP, TLOFW_ECSBS, LLOCA_ECSBS) 강화**:
  - **Tail-weighted loss**: ErrorMLP를 큰 오차에 가중(현재 SmoothL1은 큰 오차를 *덜* 반영 → 꼬리 목표와 상충). weighted-L1 / quantile / CVaR형.
  - **1-round DAgger**: 보정된 궤적에서 오차를 재수집해 ErrorMLP 미세조정(개방루프→폐쇄루프 covariate shift 완화).
  - **롤백 가드(필수)**: DAgger 라운드 후 held-out `micro_mae` **및** 게이팅 `p99`가 round-0 대비 악화되면 round-0 가중치로 롤백(무결점 tol=0). 폐쇄루프 발산 방지.
- **수용 기준(AC-remedy)**: 선택 β는 `β=0`에서 baseline과 **정확히 일치(null-op, 코드 검증됨)** → 평균 비회귀 보장. 개선 목표 = **현재 게이팅 op 대비 p99 추가 −3%p 이상**, micro·macro 평균 비회귀. op≤oracle 셀(SBO/LLOCA_ECSBS/TLOFW_CSP)은 baseline_p99 기준, TLOFW_ECSBS만 oracle-gap 유효.

### GO / NO-GO 게이트
Phase R 결과를 보고 Phase A(비싼 방법론 검증) 실행 여부를 판단.

### Phase A — Ablation 인과검증 (방법론 검증 전용, ~25 GPU-h, **게이트됨**)
- **범위**: SBO만 재학습(가장 저렴·자기완결). 노이즈 플로어(3-seed)는 더 싼 TLOFW_CSP로 대체 가능.
- **핵심 기법 — 상수-용량 블록 마스킹**: 물리 삭제 대신 채널을 train-mean으로 마스킹 → `input_size`·출력헤드 고정 → 성능 저하가 "용량"이 아닌 "정보" 손실 때문임을 귀속. **상관 형제변수(PSGGEN↔TWSG r=0.995, ZWRB(6)↔ZWRB(1) r=0.896)로 단일채널 마스크가 복원되므로 블록 단위로 제거.**
- **arm(중요도 스펙트럼)**: `sg_block`(ΔMAE합 0.116) / `zwrb_block`(0.060) / `samg02`(0.039) / `lo`(≈0) + `full`(무손상 앵커). ZWRB(6)는 1급 arm으로 포함.
- **counterfactual 삼각검증**: `mask`(삭제형) **AND** `resample`(marginal에서 추출 = permutation형) 둘 다 일치해야 인정 + `delete_capfix`(삭제+d_model 보정) 확인.
- **인과 판정(4-outcome, p99 꼬리 우선 / AR_micro는 보고용)**:
  - **PASS**: Spearman ρ(ΔMAE, damage) = 1.0 (n=4 정확 순열검정 p=1/24=0.042), mask·resample 둘 다, spread>2σ.
  - **PASS-with-caveat**: ρ=0.8(역전 1회), 모든 상위블록 damage>2σ 단조 — 정확 p=4/24=0.167을 **유의(p<0.05)가 아닌 효과크기로 정직 보고**.
  - **INCONCLUSIVE**: 잔여상관 |r|>0.9 또는 노이즈밴드≥damage 스프레드(저파워). ※ INCONCLUSIVE는 "중요도가 인과적으로 전이 안 됨" 주장으로 세탁 금지.
  - **HONEST-NEGATIVE**: 실제 무효과.
  - (참고) n=4에선 p<0.05 = 완벽순서와 동치. 유의수준 PASS를 원하면 저중요 control anchor(SAMG-06 ECSBS 0.00109, SAMG-03 0.00034) 추가 → n=5~6 (+10~15 GPU-h) — **최소 계획에는 미포함**.
- **결정 연결**: PASS→해당 센서 안전중요 표기(중복화/모니터링 권고); INCONCLUSIVE→주장 보류; HONEST-NEGATIVE→AR 강건성에 인과적 부하 없음 보고.

---

## 4. 비용 & 정직한 한계

- **Phase R ≈ 2 GPU-h** (ErrorMLP는 분 단위 학습). 저비용·고가치.
- **Phase A ≈ 25 GPU-h(SBO) / ~10(TLOFW trim)**. **방법론 검증 전용 — 어떤 강건성 지표도 개선하지 않음.**
- **정직한 예상**: Phase A는 n=4 + StepLR 저파워 + 블록간 잔여상관(~0.67)로 **INCONCLUSIVE 또는 PASS-with-caveat가 나올 확률이 높음**. 유의수준 PASS가 꼭 필요하면 anchor 추가 레버(+10~15 GPU-h)를 선택해야 함.

---

## 5. 주요 리스크(요약)
- 용량-vs-정보 혼동(→마스킹으로 해결), 재학습 노이즈 플로어(→3-seed 2σ), TF→AR 전이 격차, StepLR 붕괴(lr~1e-5@ep4로 전 arm 저파워), 형제변수 복원(→블록 ablation), DAgger 발산(→롤백 가드), oracle를 목표로 오용(→op≤oracle 셀은 baseline 기준). SBO 인과결과는 구조가 다른 타 셀로 전이 제한(SBO-scoped).

---

## 6. ADR (결정 기록)

- **Decision**: remedy-first 2단계. 게이팅 ErrorMLP를 강건화 레버로 확정하고 병목 3셀에 tail-loss+DAgger; ablation은 SBO 블록마스킹+resample 삼각검증의 방법론 검증으로 GO/NO-GO 뒤에 배치.
- **Drivers**: (1) 중요도≠오차 분리 → 변수제거는 꼬리를 못 줄임(강건성=보정기), (2) 게이팅은 이미 −10~34% 입증, (3) 비용 가시화·정직한 판정.
- **Alternatives 기각**: 전역 β(무효), 물리삭제(용량 confound), 단일채널 ablation(형제 복원), mask-only(permutation 미검증), ablation-first(비싼 likely-null 선행), n=4에서 p<0.05 주장(불가능).
- **Consequences**: Phase R 저비용 강건성 확보. Phase A는 INCONCLUSIVE 가능성 높은 방법론 검증(정직 표기).
- **Follow-ups**: 필요 시 (a) anchor 추가로 유의수준 ablation, (b) 백본 tail-loss 재학습(B2, honest-negative 게이트 뒤), (c) 타 셀 remedy 확장.

---

## 7. 실행 옵션 (사용자 결정 대기)
1. **Phase R만 먼저(권장)** — ~2 GPU-h, 강건성 즉시 확인 후 Phase A는 GO/NO-GO.
2. **Phase R + Phase A 전체** — ~27 GPU-h, Phase A는 INCONCLUSIVE 가능성 인지하고 진행.
3. **Phase A 유의수준까지** — anchor 추가(n=5~6), +10~15 GPU-h.
4. **실행 보류** — 계획만 확정, 검토 후 결정.
