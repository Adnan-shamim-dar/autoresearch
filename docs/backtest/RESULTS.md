# Replication back-test: results

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed before outcomes were joined.
Data: RPP `RPPdataConverted.xlsx`, SHA-256 `d260682aec9af5f8f7aae149fa3e017d64c53e456e9d1b11bddbf86121015e76`.

## Headline (primary rule, K = 10)

Run blind on **92** published findings, it accepted **47** and refused **45**; its accepts replicated at **49%** (95% CI 35–63%) and its refusals at **24%** (95% CI 14–39%).

Overall replication rate among included findings: 37%.

## All rules

| Rule | K | Accepted | Accepts replicated | Refused | Refusals replicated | Fisher p |
|---|---:|---:|---|---:|---|---:|
| primary | 10 | 47 | 23/47 (49%, 35–63%) | 45 | 11/45 (24%, 14–39%) | 0.018 |
| secondary | 10 | 23 | 15/23 (65%, 45–81%) | 69 | 19/69 (28%, 18–39%) | 0.0023 |
| sensitivity_k1 | 1 | 92 | 34/92 (37%, 28–47%) | 0 | 0/0 (n/a, n/a) | n/a |
| sensitivity_k5 | 5 | 57 | 23/57 (40%, 29–53%) | 35 | 11/35 (31%, 19–48%) | 0.51 |
| sensitivity_k10 | 10 | 47 | 23/47 (49%, 35–63%) | 45 | 11/45 (24%, 14–39%) | 0.018 |
| sensitivity_k20 | 20 | 39 | 21/39 (54%, 39–68%) | 53 | 13/53 (25%, 15–38%) | 0.0049 |
| sensitivity_k50 | 50 | 32 | 20/32 (62%, 45–77%) | 60 | 14/60 (23%, 14–35%) | 0.0003 |

Median prescribed N for refused findings (primary rule): 142.

## Reading this

- This is the gate's search-pricing logic applied to summary statistics, not the full engine.
- The link between small original p-values and replication was already reported by OSC (2015). This checks that a rule fixed in advance reproduces it; it is not a new finding.
- K = 10 is a convention; the sensitivity rows show how much the answer depends on it.
