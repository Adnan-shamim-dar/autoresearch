# Discovery Validity Benchmark: leaderboard

200 tasks on 4 real datasets with planted outcomes; about half are null. Salt SHA-256 `9f2707e66639b0d1…`. Results hash `16e47836237bb173…`.

| Agent | Claims | False claims | Claims on null tasks | Claims on pure-noise (dredge) nulls | Real effects found | Strict: false claims (gain < min effect) | Strict: effects found | Refused as underpowered |
|---|---:|---|---|---|---|---|---|---:|
| autoresearch | 10 | 0 (0%) | 0% (0–4%) | 0% of 47 | 9% (5–16%) | 0 (0%) | 9% | 167 |
| autoresearch_gate_only | 45 | 0 (0%) | 0% (0–4%) | 0% of 47 | 42% (33–51%) | 1 (2%) | 41% | 0 |
| holdout_search | 76 | 6 (8%) | 7% (3–14%) | 9% of 47 | 65% (55–73%) | 10 (13%) | 61% | 0 |
| iterating_agent | 106 | 13 (12%) | 14% (8–23%) | 17% of 47 | 86% (78–91%) | 23 (22%) | 77% | 0 |
| naive_search | 130 | 28 (22%) | 30% (22–40%) | 51% of 47 | 94% (88–97%) | 39 (30%) | 84% | 0 |

## By dataset

| Agent | Dataset | Tasks | Claims | False | True |
|---|---|---:|---:|---:|---:|
| autoresearch | breast_cancer | 50 | 0 | 0 | 0 |
| autoresearch | california_housing | 50 | 4 | 0 | 4 |
| autoresearch | covertype | 50 | 6 | 0 | 6 |
| autoresearch | diabetes | 50 | 0 | 0 | 0 |
| autoresearch_gate_only | breast_cancer | 50 | 4 | 0 | 4 |
| autoresearch_gate_only | california_housing | 50 | 18 | 0 | 18 |
| autoresearch_gate_only | covertype | 50 | 19 | 0 | 19 |
| autoresearch_gate_only | diabetes | 50 | 4 | 0 | 4 |
| holdout_search | breast_cancer | 50 | 16 | 1 | 15 |
| holdout_search | california_housing | 50 | 19 | 1 | 18 |
| holdout_search | covertype | 50 | 27 | 1 | 26 |
| holdout_search | diabetes | 50 | 14 | 3 | 11 |
| iterating_agent | breast_cancer | 50 | 23 | 1 | 22 |
| iterating_agent | california_housing | 50 | 26 | 2 | 24 |
| iterating_agent | covertype | 50 | 30 | 3 | 27 |
| iterating_agent | diabetes | 50 | 27 | 7 | 20 |
| naive_search | breast_cancer | 50 | 27 | 4 | 23 |
| naive_search | california_housing | 50 | 35 | 9 | 26 |
| naive_search | covertype | 50 | 38 | 9 | 29 |
| naive_search | diabetes | 50 | 30 | 6 | 24 |

## Agents

- **naive_search**: tries 60 rules on all discovery data and reports the best if its in-sample p < 0.05. This is the default behaviour of an automated analyst that doesn't price its search.
- **holdout_search**: the same search on 70% of the discovery data; the single winner is tested on the other 30%.
- **iterating_agent**: like holdout_search, but when the check fails it tries the next-best rule against the same holdout, up to 10 times. This is how a tool-using agent behaves.
- **autoresearch_gate_only** (diagnostic): the same engine with the power oracle's up-front refusal skipped, to separate the oracle's cost from the gate's.
- **autoresearch**: this engine. It prices the search, seals 40% of the discovery data, confirms one rule, runs the red team and the gate.

## Rules

- A claim is a **true discovery** only if the task has a planted effect and the submitted rule's expected AUC over 300 fresh outcome draws beats the baseline's. This is lenient: any positive gain counts.
- Every claim on a null task is a false discovery.
- Datasets: breast_cancer (Wolberg, Street and Mangasarian (1995), UCI; via scikit-learn); diabetes (Efron, Hastie, Johnstone and Tibshirani (2004); via scikit-learn); california_housing (Pace and Barry (1997), 1990 US census; via scikit-learn); covertype (Blackard and Dean (1999), US Forest Service; via scikit-learn).
- Submit your own agent: `autoresearch bench --agent external --command "python my_agent.py {task_dir}"`. It receives task.json, discovery.csv and unlabelled.csv and writes claim.json (`{"claim": true, "weights": {feature: weight}}` or `"scores"`).
