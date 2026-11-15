# Discovery Validity Benchmark

**How often does an automated discovery agent claim something that isn't there?**

Every task is a real dataset whose outcome is drawn from a known model, so the truth is known. About half the tasks are null. The agent gets the discovery rows with outcomes, plus the remaining rows without them, and returns a claim and a rule. The benchmark scores the rule against the planting model, never against the agent's own report.

```bash
autoresearch bench --tasks 200                                         # the reference agents
autoresearch bench --command "python my_agent.py {task_dir}"          # your agent, any language
```

An external agent receives `task.json`, `discovery.csv` and `unlabelled.csv` in `{task_dir}`, and writes `claim.json`: `{"claim": true, "weights": {feature: weight}}` or per-unit `"scores"`. A working example is in [examples/bench/example_agent.py](../examples/bench/example_agent.py).

## Results: development set (salt `public-dev-v1`, 200 tasks)

The full table is in [bench/LEADERBOARD.md](../bench/LEADERBOARD.md); every row is in `bench/results.json`.

| Agent | Claims on null tasks | …on pure-noise tasks | Real effects found |
|---|---:|---:|---:|
| Naive search (best of 60 rules, in-sample p < 0.05) | **30%** | **51%** | 94% |
| Holdout (search 70%, test the winner once on 30%) | 7% | 9% | 65% |
| Iterating agent (re-tests on the same holdout up to 10 times) | **14%** | 17% | 86% |
| AutoResearch | 0% (0–4%) | 0% | **9%** |
| AutoResearch without the power oracle's up-front refusal (diagnostic) | 0% (0–4%) | 0% | 42% |

On the two 3,000-row datasets:

| Agent | False claims on 45 null tasks | Effects found in 55 |
|---|---:|---:|
| Naive search | 18 | 55 |
| Holdout | 2 | 44 |
| Iterating agent | 5 | 51 |
| AutoResearch | 0 | 10 |
| AutoResearch, diagnostic | 0 | 37 |

## Pre-registered evaluation on fresh tasks

The development set below was used to diagnose the engine, so the fix was judged on 300 new tasks drawn from a secret seed whose hash was committed first. All four pre-registered hypotheses passed: [bench_eval/RESULTS.md](bench_eval/RESULTS.md). The headline: with `if_underpowered: proceed`, AutoResearch found 41% of real effects (78% on 3,000-row data) with **0 false claims in 142 null tasks**. An agent re-testing its holdout made false claims on 10.6% of them; the same holdout used once, 2.8%.

## What this shows (development set)

1. **The problem is large.** Given an outcome unrelated to anything measured, a search that doesn't price itself reports a discovery about half the time.
2. **Agents make a correct procedure unsafe.** A holdout tested once is close to its nominal error rate (7%). Let the analyst re-test against the same holdout after each failure, which is what a tool-using agent does, and the false-claim rate doubles (14%). Sealing exists to block exactly this: the vault answers once.
3. **AutoResearch made no false claims in 200 tasks, but by default it refuses most of them.** *Correction:* the rest of this point originally blamed an inflated variance estimate. Measurement showed the estimate is accurate (predicted/realised SE 1.12 and 1.03); the refusals are the spec's policy of not spending sealed data on an underpowered question, now the attributed contract choice `decision.if_underpowered`. Original text: Its power oracle refused 167 of 200 tasks as underpowered for the declared 0.05 effect. Many of those effects were real and larger than 0.05. With that refusal skipped, the same gate finds 42% of real effects (67% on the larger datasets), still with no false claims. The oracle estimates its variance from an equal-weight combination of all features, which over-states the noise of a realistic winning rule. **Calibrating the power oracle is the most valuable engineering task now open**, and this benchmark measures it.
4. **Small data is hard for everyone.** On 442–569 rows, every agent that keeps its error rate low finds few effects.

## Rules

- A claim is a **true discovery** only if the task has a planted effect and the submitted rule's expected AUC gain over the baseline, over 300 fresh outcome draws, is positive. The strict columns require the asked-for 0.05.
- Every claim on a null task is false.
- **A non-answer is not a correct silence.** If an agent crashes, declines, or spends its turn budget without submitting, the task is recorded as `no_response` and drops out of both rate denominators; the leaderboard's first column is how many tasks the agent answered. An unanswered task is no evidence either way, and scoring it as silence would hand the best false-positive rate to the agent that works least. Read the rates next to the answered count: 0% on 3 answered tasks is not a result. *(This rule was added on 2026-11-15, after a real-LLM pilot was credited with correct silences on tasks where it had simply run out of turns. The pre-registered 300-task evaluation contains no non-responses — the reference agents always answer — so none of its published numbers change; `autoresearch bench --rebuild` re-scores saved rows and confirms it.)*
- Two task families: *improve* (the baseline carries real signal; does a rule add to it?) and *dredge* (the baseline is uninformative; under the null nothing predicts the outcome).
- Datasets: Wisconsin breast cancer (569), diabetes (442), California housing (3,000-row samples of 20,640), forest covertype (3,000-row samples of 581,012). All are bundled with or fetched by scikit-learn.

## Why results can be trusted

- **No contamination.** Outcomes are re-drawn per task. Memorising a public dataset, or having trained on it, doesn't help.
- **Commit-reveal.** For an official run the task salt is kept secret, its SHA-256 is published first, and the salt is revealed after submissions close. Nobody can tune to the test tasks.
- **Deterministic.** The same salt gives the same tasks and scores on any machine. Results carry a hash.

## Limitations

- The reference agents are simple programs, not LLM agents. The point of the external-agent protocol is to run real ones.
- Planted outcomes are logistic in rank-normalised features, which every agent's rule grammar can express.
- This development set was also used to diagnose the engine. Any engine change must be evaluated on a new salt.
