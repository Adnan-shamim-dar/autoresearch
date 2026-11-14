"""Minimal external agent for the Discovery Validity Benchmark.

Usage: autoresearch bench --command "python examples/bench/example_agent.py {task_dir}"

It reads task.json and discovery.csv and writes claim.json. Replace the logic
with your own agent, which can call an LLM or anything else. This one tests a
single pre-chosen rule (the feature most correlated with y in discovery) on a
30% holdout, once.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

task_dir = Path(sys.argv[1])
spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
disc = pd.read_csv(task_dir / "discovery.csv")
rng = np.random.default_rng(0)
idx = rng.permutation(len(disc))
cut = int(len(disc) * 0.7)
search, test = disc.iloc[idx[:cut]], disc.iloc[idx[cut:]]

base = spec["baseline"]["column"]
others = [f for f in spec["features"] if f != base]
corr = {f: stats.spearmanr(search[f], search["y"]).statistic for f in others}
best = max(corr, key=lambda f: abs(corr[f]))
weights = {base: 1.0, best: float(np.sign(corr[best]))}


def auc(y, s):
    return stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())


def score(df):
    return sum(w * stats.rankdata(df[f]) for f, w in weights.items())


y = test["y"].to_numpy()
gain = auc(y, score(test)) - auc(y, stats.rankdata(test[base]))
(task_dir / "claim.json").write_text(json.dumps({"claim": bool(gain > spec["min_effect"]), "weights": weights}), encoding="utf-8")
