# Operating characteristics of the certification gate

Generated: 2026-11-14T04:36:13Z
Substrate: immunotherapy synthetic planted-signal fixture

`Detected` is the promotion rate: the fraction of replicates the gate affirmed at either the **confirmed** or **supported** tier. `Confirmed` is the stricter top-tier-only rate. `FPR (alpha)` is the fraction of pure-null replicates with a positive confirmation delta and one-sided p <= alpha; `FPR (decision)` is the fraction the gate actually promoted.

| True effect | n | Mean realised effect | Detected | 95% CI | Confirmed | FPR (alpha) | FPR (decision) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 20 | 0.001 | 0.00 | 0.00-0.16 | 0.00 | 0.00 | 0.00 |
| 0.030 | 20 | 0.030 | 0.00 | 0.00-0.16 | 0.00 | 0.00 | 0.00 |
| 0.060 | 20 | 0.060 | 0.10 | 0.03-0.30 | 0.00 | 0.00 | 0.00 |
| 0.100 | 20 | 0.100 | 0.85 | 0.64-0.95 | 0.00 | 0.00 | 0.00 |
| 0.150 | 20 | 0.150 | 1.00 | 0.84-1.00 | 0.85 | 0.00 | 0.00 |

## Setup

- Replicates per effect: 20
- Candidate bank: 32
- Selection-null permutations: 39
- Discovery patients: 480
- Confirmation groups: 160
- Wall clock: 365.9 s

## Method

Replicates use the real contract validator, proposer bank, group-aware CV, selection-aware null, sealed-set aggregate confirmation, power oracle and gate. Effects are planted in a known latent score and realised on the generated cohort; the search never sees that score.

## Limitations

- Synthetic substrate: this table demonstrates the gate's operating characteristics; it is not validated instrument performance on real data.
- Adversary passes are held at pass to isolate gate behaviour; the adversary's own false-alarm rate is reported separately when it is calibrated.
- Effects are planted in a known latent score; the search sees only the contract features, never the planted score.
- A refusal caused by the power oracle is counted as a non-detection, because that is what the pipeline would actually emit.
