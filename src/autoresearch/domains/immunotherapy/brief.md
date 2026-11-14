# Domain brief — personalized immunotherapy

This is the first application domain for AutoResearch. It is a domain where the
question shapes recur constantly and where the difference between a
reproducible claim and an expensive-looking correlation is measured in patient
cohorts and in capital.

## The clinical benchmark

Individualized neoantigen mRNA vaccines (intismeran autogene / mRNA-4157 plus
pembrolizumab) reached a Phase 3 recurrence-free survival win in resected
melanoma in August 2026. The construct encodes up to 34 patient-specific
neoantigens, is manufactured per patient, and takes roughly 4–8 weeks from
biopsy to dosing. Reported unit costs exceed $300,000 per patient and
press-cited prices approach $475,000. The Phase 3 overall-survival data remain
immature, treatment discontinuation from adverse events was 14.0% versus 8.7%
on pembrolizumab alone, and cold tumours remain uneven.

Competing modalities are not merely different products; they are different
*pricing structures for the same T-cell problem*:

- **Autologous TIL (lifileucel/Amtagvi).** ~34-day manufacture, $562,000 list
  price, 49% real-world ORR in melanoma, 56% reported gross margin (Q2 2026),
  meaningful toxicity and a withdrawn EU application.
- **Engineered TCR-T (afamitresgene autoleucel/Tecelra).** Shared MAGE-A4
  antigen restricted to selected HLA-A*02 subtypes; ~6-week manufacture;
  biomarker testing gates eligibility; CRS 72% (2% grade ≥3).
- **Off-the-shelf soluble TCR bispecific (tebentafusp/Kimmtrak).** gp100/HLA-
  A*02:01 × anti-CD3; 11% ORR, 21.6-month median OS in uveal melanoma; 99.6%
  reported gross margin (Q1 2026) because there is no per-patient batch.
- **T-cell engagers / bispecific antibodies (tarlatamab, amivantamab).** Surface
  targets, zero batch delay, CAD ~30.9K–32.4K per 28-day cycle for tarlatamab.
- **Solid-tumour CAR-T (CLDN18.2, CD133, CEA).** Investigational; encouraging
  ORR signals and substantial on-target toxicity.

The economic fault line is not efficacy alone. It is **N=1 manufacturing versus
off-the-shelf batch economics**. A discovery engine that can price a search and
refuse an underpowered comparison is directly useful here: the field is full of
small cohorts, surrogate endpoints, and modality comparisons that are asserted
more often than they are identified.

## The disease expansion

The same computational stack—epitope mapping, MHC-I/II binding prediction,
TCR–pMHC matching, mRNA/LNP or vector delivery—extends beyond oncology:

- **Autoimmune disease.** Inverse/tolerogenic vaccines deliver gluten peptides
  (celiac, KAN-101), MBP/MOG peptides (MS), or insulin peptides (T1D) without
  co-stimulation, aiming at antigen-specific tolerance rather than chronic
  immunosuppression. B-cell-depleting CAR-T (miv-cel in stiff person syndrome)
  is in a rolling BLA with a Q4 2026 target. The 2025 biopharma market saw 106
  autoimmune deals, second only to oncology.
- **Chronic viral infection.** Chronic HBV affects >240M people; bepirovirsen
  was approved in October 2026 with 19–20% functional cure versus 0% placebo,
  leaving ~80% uncured. Therapeutic mRNA vaccines (BNT168 in HIV, Luye in HPV)
  and mRNA-LNP + nuclease combinations are in clinical development.
- **Allergy and bacterial/AMR pathogens.** Epitope mapping, allergenicity
  scoring, population-HLA polytope design, and resistance profiling are
  computational pipelines without the same clinical readouts.

## The data landscape

The field splits cleanly into:

1. **Immunogenicity-labelled resources** — CEDAR, IEDB, TANTIGEN 2.0, TESLA,
   VDJdb, McPAS-TCR, ImmuneCODE, 10x/NetTCR, and the validated tiers of
   dbPepNeo2.0, NEPdb, and TSNAdb v2.0. These carry experimental T-cell,
   B-cell, or TCR–pMHC outcomes. They are the only substrates on which a
   discovery claim about immune recognition can be confirmed.
2. **Unlabelled resources** — TCGA/TARGET/GENIE, CPTAC, SysteMHC Atlas,
   OGRDB/VDJbase, AlphaFoldDB, and the predicted tiers of NEPdb/TSNAdb. These
   are discovery or benchmark inputs, not confirmation sets. An engine that
   confuses presentation with immunogenicity will certify a beautiful artifact.

Access matters: TESLA sits behind Synapse authentication; raw GDC files require
controlled access; DTU standalone tools require commercial licences (with
ImmunoGeNN as a documented exception); clinical use of an epitope-selection
pipeline triggers SaMD/CDS and EU IVDR/AI Act questions.

## The first runnable question

**Does a multi-feature within-patient ranking rule beat a binding-only baseline
at putting validated neoantigens in the top 3, on a held-out patient cohort?**

- Unit: neoepitope.
- Group: patient.
- Outcome: validated T-cell recognition (binary).
- Baseline: NetMHCpan-style binding score alone.
- Candidate space: weighted rank fusion of binding, stability, expression,
  foreignness, presentation, self-similarity, processing, and publicness.
- Metric: paired delta AUC; secondary top-3 enrichment.
- Effect that matters: 0.03 AUC.
- Decision: advance the panel to prospective validation only if the held-out
  effect clears 0.03 with the search price paid.

The demo substrate in this repository is synthetic and TESLA/CEDAR-shaped. It
exists to exercise every module end to end. The contract is the same one you
would point at the real TESLA or CEDAR export once access is granted; only the
data hash changes.

## The economics question that must be refused

**Are off-the-shelf modalities more profitable than autologous N=1 products?**

The report contains two usable gross-margin observations: 56.1% for TIL and
99.6% for an off-the-shelf soluble TCR. Two observations across different
products and disease settings cannot certify a 10-point margin difference. The
correct output is a refusal with a sample-size prescription, not a chart with a
line through two points. This is a first-class demonstration of the product.

## What would make a claim portable

1. The operating-characteristics table on the real substrate, not a simulation.
2. One correct yes that an independent cohort agreed with.
3. One correct no where a conventional analysis would have published.
4. A replication back-test against known findings.
5. Third-party reproduction from the certificate alone.

Until then, AutoResearch is an instrument under calibration, not an oracle.
