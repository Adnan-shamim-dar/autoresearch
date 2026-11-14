# Results: `if_underpowered: proceed` on fresh benchmark tasks

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), commit 30bb4dc, made before the run.
Salt, now revealed: `eval-2026-11-14-03b53ae2cf9bf59ddc69c2311cf9a5d2`. Its SHA-256 matches the committed hash (`f13450b2…`).
Full rows: [bench/eval/results.json](../../bench/eval/results.json); leaderboard: [bench/eval/LEADERBOARD.md](../../bench/eval/LEADERBOARD.md).

## Hypotheses, scored as written

| | Rule | Result | Verdict |
|---|---|---|---|
| H1 | `autoresearch_proceed` claims on ≤ 5% of null tasks | 0 of 142 (95% CI 0–2.6%) | **Pass** |
| H2 | `autoresearch_proceed` finds ≥ 50% of real effects on the 3,000-row datasets | 56 of 72 (78%) | **Pass** |
| H3 | `iterating_agent` makes more null-task claims than `holdout_search` (one-sided Fisher p < 0.05) | 15 vs 4 of 142, p = 0.008 | **Pass** |
| H4 | `naive_search` claims on ≥ 30% of pure-noise null tasks | 22 of 68 (32%) | **Pass** |

## All agents (300 tasks)

| Agent | Claims on null tasks | Real effects found | On 3,000-row data | False claims / claims |
|---|---:|---:|---:|---:|
| naive_search | 16.9% | 96% | 72/72 | 25/177 |
| iterating_agent | 10.6% | 82% | 71/72 | 15/145 |
| holdout_search | 2.8% | 68% | 63/72 | 4/112 |
| autoresearch (refuse, the default) | 0% | 10% | 16/72 | 0/16 |
| **autoresearch_proceed** | **0%** | **41%** | **56/72** | **0/64** |

## Reading this honestly

- **The fix works as intended.** With the refusal turned into an attributed choice, the engine finds four times as many real effects (41% vs 10%) and still makes no false claims in 300 tasks.
- **A one-shot holdout is a strong competitor when nobody peeks.** It finds more effects (68%) at a 2.8% false-claim rate. AutoResearch's case rests on two things the holdout lacks. First, it holds up when the analyst iterates: re-testing the same holdout raised false claims to 10.6% here (H3). The vault answers once, so that can't happen. Second, it produces a signed record of what was searched and spent.
- **Small data remains hard.** On the 442–569-row datasets, every agent that keeps its error rate low finds few effects.
- **The size of the problem varies between task sets.** Naive search claimed on 51% of pure-noise tasks on the development set and 32% here. Quote both, or the pre-registered 32%.
