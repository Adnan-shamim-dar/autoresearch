# Pre-registration: evaluating `if_underpowered: proceed` on fresh benchmark tasks

Committed to git **before** the run. The task salt is secret until the run finishes; only its hash is published here:

`sha256(salt) = f13450b2758261bf6bb356edcb8e78fb697a0caa3e66701b4577efe4ded0f431`

## Why

On the development set (`public-dev-v1`) AutoResearch made no false claims but found only 9% of real effects. A diagnostic on that same set showed the power oracle's variance estimate is accurate (median ratio of predicted to realised confirmation SE: 1.12 on *improve* tasks, 1.03 on *dredge* tasks). The low power is a policy, not an estimator bug: the engine refuses whenever the declared 0.05 effect is below the minimum detectable effect (0.07–0.10 at these sizes), even when the true effect is larger. The fix is a contract option, `decision.if_underpowered: refuse | proceed`, with refuse as the default, and proceed has to name who chose it. Because the development set was used for this diagnosis, the evaluation runs on new tasks.

## Design

- Salt: secret (hash above), revealed in `bench/eval/results.json` after the run.
- 300 tasks, generated exactly as on the development set (same code, same parameters).
- Agents: `naive_search`, `holdout_search`, `iterating_agent`, `autoresearch` (refuse), `autoresearch_proceed`.
- Command: `autoresearch --root bench-eval-root bench --tasks 300 --salt <salt> --agent ...` (output copied to `bench/eval/`).

## Hypotheses and pass rules (fixed now)

- **H1, safety (primary).** `autoresearch_proceed`'s claim rate on null tasks is **≤ 5%** (point estimate). The Wilson 95% CI is reported too.
- **H2, power.** `autoresearch_proceed` finds **≥ 50%** of real effects on the 3,000-row datasets (California housing, covertype).
- **H3, the agentic-reuse finding replicates.** `iterating_agent`'s null-task claim rate is higher than `holdout_search`'s (one-sided Fisher exact test, p < 0.05).
- **H4, the size of the problem replicates.** `naive_search` claims a discovery in **≥ 30%** of pure-noise (*dredge*) null tasks.

Every hypothesis is reported pass or fail as written, whatever the result. No re-runs with other salts.
