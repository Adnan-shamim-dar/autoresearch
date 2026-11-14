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
- The sales package was written outside the repo: `C:\Users\rambe\Desktop\autoresearch-sales\`.

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
