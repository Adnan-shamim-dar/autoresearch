# Project log

A dated record of what was built, what was decided and why, and what the results were, including mistakes. Newest entries at the bottom. Commit hashes refer to this repository.

## 2026-11-14: origin

- DeepSeek (in opencode) started AutoResearch from the build specification `Auto.md` and the immunotherapy report, then hit its usage limit. Claude Code continued from the opencode chat history.
- The user's standard: follow `Auto.md` "to the dot". `docs/SPEC_COVERAGE.md` tracks status section by section.

## 2026-11-14: spec build-out (commits up to fdb9f7c)

- Implemented the remaining spec items: physical sealing (OS lock plus a separate label-service process), procedure registration, discovery-only selection-aware null, pseudo-group control, private refusals, operator log, refusal taxonomy, α dial, interpretability bill, an MCP `verify` tool, the web console, and a second real-data domain (WDBC).
- Result: 125 tests. Synthetic OC table: 0/20 false positives. Feedback-leak test: a leaky channel promoted 36/100 noise runs, the guarded channel 1/100.

## 2026-11-14: "what's missing to sell it"

- Answer: real-data evidence (a real-data OC table, a replication back-test, third-party reproduction), a public home for the code, and a licence.
- Decision: prepare everything, but **send nothing to anyone without the user's explicit OK per batch**.

## 2026-11-14: replication back-test (0a91747, 302abdb)

- The pre-registration (rule, inclusion, outcome, analysis) was committed in 0a91747 **before** any replication outcome was joined.
- Data: Reproducibility Project: Psychology, `RPPdataConverted.xlsx`, SHA-256 `d260682a…`.
- Result: 92 findings. Accepted 47 (49% replicated), refused 45 (24% replicated), Fisher p = 0.018. The stricter rule gave 65% vs 28%. **Not significant at K = 5** (p = 0.51); this is reported in the results.

## 2026-11-14: real-data OC table and reproduction kit (b42f145)

- Plasmode design on 569 real WDBC tumours, 100 replicates per effect: 0/100 null promotions; detection 0% up to a true AUC gain of 0.10, 44% at 0.15. The binding check was the selection-aware null.
- Bug found and fixed: in calibration only, for no-group data, the leakage scan compared `__all__` with `__all__` and flagged an overlap that wasn't there. The engine itself was unaffected.
- `scripts/reproduce_proofs.py` recomputes the evidence from public data. It passed from a clean folder.
- The sales package was written in a separate folder outside the repo.

## 2026-11-14: "do we have anything impressive?" → Discovery Validity Benchmark (eb7a96f)

- Honest assessment: careful engineering of known methods; nothing a payer couldn't rebuild. The missing asset was a **measurement**.
- Built a benchmark on 4 real datasets with outcomes re-drawn from a hidden model. It can't be contaminated, supports commit-reveal official runs, and any agent can enter through a folder in and `claim.json` out.
- First version mistake: the task type "beat a strong baseline" barely fooled anyone (naive search 0/25 false claims). Added *dredge* tasks (an uninformative baseline; under the null nothing predicts the outcome), which is how real false discoveries happen. Raised the minimum effect to 0.05.
- Development-set results (salt `public-dev-v1`, 200 tasks): naive search claimed on 51% of pure-noise tasks. Holdout 7% null claims; iterating agent 14%. AutoResearch 0% null claims but **found only 9% of real effects**; with the oracle's refusal skipped as a diagnostic, 42%.

## 2026-11-14: power oracle investigation (this entry is a correction)

- **Wrong diagnosis, now corrected:** the docs at eb7a96f say the oracle "over-states the noise" because its reference candidate is an equal-weight combination of all features. Measured on 80 development tasks, the oracle's SE is accurate: median ratio of predicted to realised confirmation SE was 1.12 (improve) and 1.03 (dredge).
- Real cause: a **policy**. The engine refuses whenever the declared smallest effect (0.05) is below the minimum detectable effect (0.07–0.10 at these sizes), even if the true effect is larger.
- Fix: the contract option `decision.if_underpowered: refuse | proceed`. The default is refuse, as the spec says. Proceed must name who chose it, the certificate records it, and the certificate states that a null result is not evidence of absence. Error control is unchanged: the gate is the same.
- The evaluation is pre-registered on a fresh secret salt (`docs/bench_eval/PREREGISTRATION.md`), because the development set was used for the diagnosis.

## Standing decisions waiting on the user

- A public GitHub repo (irreversible once public), the licence (pyproject says Apache-2.0; no LICENSE file yet), a name and contact for the materials, and a go-ahead per outreach batch.
- `/compact` is a command the user types; Claude can't trigger it.

## 2026-11-14: pre-registered evaluation of `if_underpowered: proceed` (scored)

- The salt hash was committed in 30bb4dc before the run. The salt is now revealed: `eval-2026-11-14-03b53ae2cf9bf59ddc69c2311cf9a5d2`; its hash matches.
- 300 fresh tasks, 5 agents. **All four hypotheses passed:** H1 0/142 null claims; H2 56/72 effects on 3,000-row data; H3 iterating 15/142 vs holdout 4/142 (p = 0.008); H4 naive 22/68 pure-noise claims (32%, down from 51% on the dev set).
- Honest caveat recorded in `docs/bench_eval/RESULTS.md`: a one-shot holdout that nobody peeks at finds more effects (68%) at 2.8% false claims. AutoResearch's edge is resistance to re-testing (the vault answers once) and the signed record.

## 2026-11-14: LLM agent adapter and public-key signatures

- `examples/bench/llm_agent.py`: a Claude analyst (`claude-opus-5-5`, effort medium, manual tool loop) with the tools `evaluate_rule` and `submit_claim`. Server-side refusal fallback is on (`fallbacks: "default"`). The tool logic is tested locally. **The API path is untested: there is no key on this machine.** Each task costs money, so start with about 40 tasks.
- Certificates now carry an Ed25519 public-key signature next to the HMAC one. The key is derived from the existing secret, and signatures are deterministic, so certificates stay byte-reproducible. Verify with `autoresearch public-key`. `cryptography` is an optional dependency (`pip install -e ".[sign]"`). It was installed into the project's Python 3.14; `pip` on PATH points at a different Python (3.11).

## 2026-11-15: release-readiness pass

- Tested the buyer's first experience: fresh clone, clean virtual environment, `scripts/reproduce_proofs.py`. **It failed**: the back-test check needed `openpyxl`, which was installed on the dev machine but never declared. Fixed in `pyproject.toml`; a second fresh clone passes all four checks.
- Added `LICENSE` (Apache-2.0, matching what `pyproject.toml` already declared; the user can still change it before publishing) and a GitHub Actions workflow (tests and the reproduction kit on Linux and Windows). The workflow has never run: there is no remote yet.
- Status: release-ready for publishing. Not "perfect": no outside reproduction yet, no real LLM agent measured yet, and one-shot holdouts still find more effects than the engine does.

## 2026-11-15: any-dataset path, real-LLM pilot, publication

- User instruction: finish the job completely, take identity from the portfolio and the model key from opencode, publish, and do the outreach. This replaces the earlier per-batch approval rule.
- **Custom domain** (`init --domain custom`): any CSV with a binary outcome and a baseline column becomes a sealed, contracted run in one command. Tested end to end on housing data (non-medical) and on a grouped dataset. Limits stated in the README: binary outcomes and numeric features only.
- **OpenRouter agent** (`examples/bench/openrouter_agent.py`): the opencode credential is a free-tier OpenRouter key (free models only, small daily cap), so only a small pilot is possible. The key is read from the environment at run time and is never written to the repository.
- Pre-publication scan of all tracked files and history: no keys, passwords or personal paths (one local path removed from this log).
- Publishing: GitHub refused the first push because the login token lacks the `workflow` scope. The workflow file was removed from the three newest commits (the pre-registration commits 0a91747 and 30bb4dc are older and unchanged) and kept as a template at `docs/ci/github-actions.yml`. To enable it: `gh auth refresh -s workflow`, then copy it to `.github/workflows/ci.yml`.
- Published: https://github.com/Adnan-shamim-dar/autoresearch (public, Apache-2.0). A clone from GitHub was checked.
- Outreach, first batch, sent by email on the user's instruction: Institute for Replication and Center for Open Science (independent reproduction requests), FutureHouse (offer to benchmark their agent). Replies and follow-ups are tracked outside the repository.
- Real-LLM pilot (pipeline check only, not evidence): 8 tasks, `nvidia/nemotron-3.5-super-vl:free` through OpenRouter. The model ran 8-18 rule evaluations per task. It hit the turn limit without submitting on 4 tasks, the last task failed on the free tier's rate limit, and of the 3 it finished it made one true discovery and one false claim on a pure-noise task. Eight tasks say nothing statistically. The pilot shows the external-agent pipeline works with a real model and that the turn limit (8) is too low; a meaningful run needs paid credits and about 200 tasks.

## 2026-11-15 (later): a scoring defect the pilot exposed

The pilot's own result table is what found this. Two of the four tasks the model abandoned at the turn limit were printed as `correct_silence`, because the scorer treated any non-claim on a null task as a deliberate silence. Under that rule an agent that crashes on every task scores a perfect false-discovery rate. The pilot's "1 true discovery, 1 correct silence, 1 false claim" line was itself produced by the flawed rule.

Fixed in `bench.py`:

- A non-claim whose `detail` carries an `error`, or a `stopped` reason of `turn limit`, `no submission` or `refusal`, is now labelled `no_response`. It is not a decision, so it is not scored as one.
- `summarise()` computes the null-task claim rate and power over **answered** tasks only, and reports `answered`, `no_response` and `response_rate`. An unanswered task is no evidence either way; keeping it in the denominator would reward failure.
- The leaderboard's first column is now the answered count, and both the Markdown rules section and the CLI line print it, so a clean rate on three answered tasks cannot be read as a clean rate.
- `autoresearch bench --rebuild` re-scores saved rows under the current rules without running an agent. `rebuild_leaderboard` existed but had no CLI path; now it does.

**The pre-registered 300-task evaluation is unchanged.** Its 1,500 rows contain zero non-responses (the reference agents always answer), and re-scoring them with `--rebuild` reproduces every published number exactly: null-task claim rates 0%, 0%, 2.8%, 10.6%, 16.9% and power 10.1%, 40.5%, 68.4%, 82.3%, 96.2%. This is stated because changing a scoring rule after pre-registration is precisely the move a referee should distrust, so the check is on the record rather than asserted.

Also fixed: the default turn limit was 8 in the OpenRouter agent, which caused half the pilot's non-answers. Both example agents now default to 24 turns and read `AUTORESEARCH_LLM_MAX_TURNS`.
