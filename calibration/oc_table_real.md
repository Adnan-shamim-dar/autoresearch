# Operating characteristics of the certification gate on real data

Generated: 2026-11-14T16:29:20Z
Substrate: real data: WDBC (569 tumours), plasmode planted outcomes

`Detected` is the promotion rate: the fraction of replicates the gate affirmed at either the **confirmed** or **supported** tier. `Confirmed` is the stricter top-tier-only rate. `FPR (alpha)` is the fraction of pure-null replicates with a positive confirmation delta and one-sided p <= alpha; `FPR (decision)` is the fraction the gate actually promoted.

| True effect | n | Mean realised effect | Detected | 95% CI | Confirmed | Without adversary | Adversary kills | FPR (alpha) | FPR (decision) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 100 | 0.000 | 0.00 | 0.00-0.04 | 0.00 | 0.00 | 0 | 0.01 | 0.00 |
| 0.030 | 100 | 0.030 | 0.00 | 0.00-0.04 | 0.00 | 0.00 | 0 | 0.01 | 0.00 |
| 0.060 | 100 | 0.060 | 0.00 | 0.00-0.04 | 0.00 | 0.00 | 0 | 0.01 | 0.00 |
| 0.100 | 100 | 0.100 | 0.00 | 0.00-0.04 | 0.00 | 0.00 | 0 | 0.01 | 0.00 |
| 0.150 | 100 | 0.150 | 0.44 | 0.35-0.54 | 0.44 | 0.44 | 0 | 0.01 | 0.00 |

## Adversary false-alarm rate (spec §9)

Of 44 replicates with a real planted effect that the gate would promote without the adversary, the adversary killed 0: false-alarm rate 0.00 (95% CI 0.00-0.08).

| Attack | Runs on real effects | Fail | Warn |
|---|---:|---:|---:|
| baseline_advocate | 400 | 7 | 58 |
| feature_permutation:mean_area | 119 | 0 | 17 |
| feature_permutation:mean_compactness | 71 | 0 | 11 |
| feature_permutation:mean_concave_points | 236 | 0 | 18 |
| feature_permutation:mean_concavity | 75 | 0 | 19 |
| feature_permutation:mean_fractal_dimension | 24 | 0 | 12 |
| feature_permutation:mean_perimeter | 145 | 0 | 40 |
| feature_permutation:mean_radius | 135 | 0 | 29 |
| feature_permutation:mean_smoothness | 100 | 0 | 8 |
| feature_permutation:mean_symmetry | 25 | 0 | 7 |
| feature_permutation:mean_texture | 154 | 0 | 3 |
| group_label_permutation | 400 | 0 | 400 |
| leakage_scan | 400 | 0 | 0 |
| leave_one_site_out | 400 | 0 | 0 |
| outcome_permutation | 400 | 45 | 0 |
| pseudo_group | 400 | 0 | 0 |

## Setup

- Replicates per effect: 100
- Candidate bank: 32
- Selection-null permutations: 39
- Discovery patients: 569 tumours (no groups)
- Confirmation groups: 40% stratified split, re-drawn per replicate
- Wall clock: 98.1 s

## Method

Operating characteristics on real data (spec §7 and §14, proof 1).

Plasmode design: the 569 real WDBC tumours keep their real 30 measurements,
their real joint distribution, correlations and tails, and the real sample
size. Only the outcome is re-drawn, so the true effect is known:

    logit P(y = 1) = a + s * (z_baseline + beta * z_extra)

``z_baseline`` is the deployed baseline (tumour size, ``mean_radius``),
``z_extra`` the mean of three other real measurements. With ``beta = 0`` the
baseline is the whole truth and any rule that "beats" it is a false positive.
``beta`` is bisected per replicate so the realised AUC gain over the baseline
matches the requested row. Each replicate also re-draws the sealed split.

The engine never sees ``z_extra`` or ``beta``: it gets the contract's ten
features and has to find the signal.

## Limitations

- Plasmode design: real measurements, real correlations and the real sample size (569 tumours); the outcome is re-drawn from a known model so the true effect is known. Real outcomes cannot carry a planted effect of known size.
- One real dataset, no group structure. The group-aware machinery is calibrated only on the synthetic table.
- The planting model is logistic-linear in rank-normalised features, which the rule grammar can express. A truth the grammar cannot express would be harder to find.
- Candidates come from the operator-family seed bank (no adaptive generations), as in the synthetic table.
- A refusal by the power oracle is counted as a non-detection.
