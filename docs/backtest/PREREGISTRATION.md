# Pre-registration: replication back-test (milestone 9)

Committed to git **before** any replication outcome was joined to any decision. The commit hash and timestamp of this file are the record.

## Question
Run blind over published findings whose replication outcomes are already known, and ask: do the findings AutoResearch's gate would **accept** replicate more often than the ones it would **refuse**?

## Data
Reproducibility Project: Psychology (Open Science Collaboration, 2015, *Science* 349:aac4716), file `data_allformats/RPPdataConverted.xlsx` from github.com/CenterForOpenScience/rpp. Its SHA-256 is recorded in the result file.

**What the decision may use.** Only information available at original publication: `T.N.O` (original N), `T.pval.USE.O` (original p-value) and `T.r.O` (original effect as a correlation).

**p-values are used as given.** `T.pval.USE.O` is the RPP's recomputed original p-value; no tail conversion is applied. The z-score for the secondary rule is z = Φ⁻¹(1 − p/2).

**Inclusion.** Rows whose `T.pval.USE.O`, `T.N.O`, `T.pval.USE.R` and `Direction.R` are all present, and whose original p < 0.05 (the original claimed a finding).

## Outcome (defined before looking)
**Replicated** means replication p < 0.05 (`T.pval.USE.R`) **and** the replication effect is in the same direction (`Direction.R == "same"`). This is the RPP's primary significance criterion.

## The gate, from summary statistics only
A published finding is the winner of an unreported search: analyses, outcomes, covariates and subgroups (the garden of forking paths). AutoResearch prices a search before it trusts a winner, so the back-test applies the same pricing with an implicit search size K.

- **Primary rule (K = 10).** Accept if and only if the original two-sided p ≤ 0.05 / 10 = 0.005, the search-priced significance threshold. Otherwise refuse.
- **Secondary rule (search-priced MDE, K = 10, 80% power).** Accept if and only if the original |z| ≥ z(1 − 0.005/2) + z(0.80) = 3.649, i.e. the claimed effect clears the minimum detectable effect after pricing the search. This is the power oracle's ANSWERABLE criterion.
- **Sensitivity (reported, not used for the headline).** The primary rule at K ∈ {1, 5, 10, 20, 50}.

**Prescription.** Each refusal states the N needed to detect the claimed r at the search-priced α with 80% power, using the Fisher z approximation: n = ((z(1 − α/2) + z(0.8)) / atanh(r))² + 3.

## Code
`src/autoresearch/backtest.py`, committed together with this file. Decisions are computed from a frame holding only the three input columns; outcomes are joined afterwards. Run with `autoresearch backtest --data RPPdataConverted.xlsx`.

## Analysis
For each rule: the number accepted and refused; the replication rate among accepts and among refusals, each with a Wilson 95% CI; the difference; and a two-sided Fisher exact test.

The headline sentence takes the form the spec asks for: *"Run blind on N published findings, it accepted X and refused Y; its accepts replicated at A% and its refusals at B%."*

## Known limitations, declared in advance
- **Not the full engine.** This is the gate's pricing logic on summary statistics, because the original raw data are not available. Contract validation, sealed confirmation and the adversary do not apply.
- **K = 10 is a convention.** It is not an estimate of each study's actual search; the sensitivity analysis shows how much the answer depends on it.
- **Known in the literature.** The relationship between smaller original p-values and replication is already reported (OSC 2015). This back-test checks that the engine's rule reproduces it with a stated, pre-committed threshold. It is not a new discovery.
- **Single field, single era.** One field (psychology), one era (studies published in 2008).
