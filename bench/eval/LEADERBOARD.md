# Discovery Validity Benchmark: leaderboard

300 tasks on 4 real datasets with planted outcomes; about half are null. Salt SHA-256 `f13450b2758261bf…`. Results hash `9e8f7818de89fb0c…`.

| Agent | Claims | False claims | Claims on null tasks | Claims on pure-noise (dredge) nulls | Real effects found | Strict: false claims (gain < min effect) | Strict: effects found | Refused as underpowered |
|---|---:|---|---|---|---|---|---|---:|
| naive_search | 177 | 25 (14%) | 17% (12–24%) | 32% of 68 | 96% (92–98%) | 52 (29%) | 79% | 0 |
| holdout_search | 112 | 4 (4%) | 3% (1–7%) | 4% of 68 | 68% (61–75%) | 19 (17%) | 59% | 0 |
| iterating_agent | 145 | 15 (10%) | 11% (7–17%) | 15% of 68 | 82% (76–87%) | 39 (27%) | 67% | 0 |
| autoresearch | 16 | 0 (0%) | 0% (0–3%) | 0% of 68 | 10% (6–16%) | 1 (6%) | 9% | 238 |
| autoresearch_proceed | 64 | 0 (0%) | 0% (0–3%) | 0% of 68 | 41% (33–48%) | 2 (3%) | 39% | 0 |

## By dataset

| Agent | Dataset | Tasks | Claims | False | True |
|---|---|---:|---:|---:|---:|
| naive_search | breast_cancer | 75 | 45 | 2 | 43 |
| naive_search | california_housing | 75 | 49 | 10 | 39 |
| naive_search | covertype | 75 | 42 | 9 | 33 |
| naive_search | diabetes | 75 | 41 | 4 | 37 |
| holdout_search | breast_cancer | 75 | 25 | 0 | 25 |
| holdout_search | california_housing | 75 | 35 | 0 | 35 |
| holdout_search | covertype | 75 | 31 | 3 | 28 |
| holdout_search | diabetes | 75 | 21 | 1 | 20 |
| iterating_agent | breast_cancer | 75 | 35 | 3 | 32 |
| iterating_agent | california_housing | 75 | 42 | 3 | 39 |
| iterating_agent | covertype | 75 | 37 | 5 | 32 |
| iterating_agent | diabetes | 75 | 31 | 4 | 27 |
| autoresearch | breast_cancer | 75 | 0 | 0 | 0 |
| autoresearch | california_housing | 75 | 10 | 0 | 10 |
| autoresearch | covertype | 75 | 6 | 0 | 6 |
| autoresearch | diabetes | 75 | 0 | 0 | 0 |
| autoresearch_proceed | breast_cancer | 75 | 3 | 0 | 3 |
| autoresearch_proceed | california_housing | 75 | 33 | 0 | 33 |
| autoresearch_proceed | covertype | 75 | 23 | 0 | 23 |
| autoresearch_proceed | diabetes | 75 | 5 | 0 | 5 |

## Agents

- **naive_search**: tries 60 rules on all discovery data and reports the best if its in-sample p < 0.05. This is the default behaviour of an automated analyst that doesn't price its search.
- **holdout_search**: the same search on 70% of the discovery data; the single winner is tested on the other 30%.
- **iterating_agent**: like holdout_search, but when the check fails it tries the next-best rule against the same holdout, up to 10 times. This is how a tool-using agent behaves.
- **autoresearch_proceed**: the same engine with the contract option `if_underpowered: proceed`: underpowered questions are run and labelled, not refused.
- **autoresearch**: this engine. It prices the search, seals 40% of the discovery data, confirms one rule, runs the red team and the gate.

## Rules

- A claim is a **true discovery** only if the task has a planted effect and the submitted rule's expected AUC over 300 fresh outcome draws beats the baseline's. This is lenient: any positive gain counts.
- Every claim on a null task is a false discovery.
- Datasets: breast_cancer (Wolberg, Street and Mangasarian (1995), UCI; via scikit-learn); diabetes (Efron, Hastie, Johnstone and Tibshirani (2004); via scikit-learn); california_housing (Pace and Barry (1997), 1990 US census; via scikit-learn); covertype (Blackard and Dean (1999), US Forest Service; via scikit-learn).
- Submit your own agent: `autoresearch bench --agent external --command "python my_agent.py {task_dir}"`. It receives task.json, discovery.csv and unlabelled.csv and writes claim.json (`{"claim": true, "weights": {feature: weight}}` or `"scores"`).
