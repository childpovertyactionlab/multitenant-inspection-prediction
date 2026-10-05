# Notebooks: testing the Chelsea inspection model on Dallas

> **Status: exploratory and unreviewed.** Notebooks 03–06 and this summary come from an
> autonomous exploration by Claude Opus 5.5 (Anthropic) on 2026-10-01, run at the repo
> owner's request without human supervision. No person has checked the feature choices,
> methods or numbers. Treat the results as leads to verify. Notebook 02 was also written by
> Claude, from a plan the repo owner reviewed and approved before it ran.
>
> Notebooks 07–11 come from a second Claude Opus 5.5 session on 2026-10-05. The repo owner
> set the questions (inspector consistency and the five follow-up analyses) and the session
> carried them out without supervision of individual steps. They are also unreviewed. That
> session also corrected notebook 06; see the note at the top of its findings.

## The question

Chelsea, MA trained a model on properties inspected proactively in one target area and
used it to rank the rest of the city for inspection. Dallas inspects every multitenant
property, so the Chelsea design can be tested with the outcome known everywhere: treat
part of the city as the only inspected area, train there, and score the rest.

The outcome throughout is a score in the bottom 10% of graded inspections (≤ 86;
FY2020-21 to FY2024-25). That is a statistical cut. Code Compliance treats 70 and above
as passing.

## Notebooks

| Notebook | Content |
|---|---|
| `01_replicate_bottom10.ipynb` | The original exercise skeleton (EDA only). |
| `02_spatial_holdout.ipynb` | Baseline. Chelsea mirror (train on the highest-poverty third of tracts, test on the rest), random 70/30 split, tract-grouped CV. LASSO, random forest, XGBoost. Self-contained. |
| `03_engineered_features.ipynb` | DCAD building systems, spatial context, owner track record, nearby outcomes. |
| `04_supplemental_sources.ipynb` | Evictions, DCAD roll history, DCTO tax roll, CoStar, DPD crime, Dallas 311, each tagged pre-window, during or snapshot. |
| `05_temporal.ipynb` | Rolling-origin design: predict each property's first inspection of a fiscal year from what was known on 1 October. |
| `06_outcome_noise.ipynb` | How much of a score depends on the inspector; inspector-adjusted target; regression against classification. |
| `07_inspection_records.ipynb` | Inspector identity across Salesforce and Code's backfill: coverage, consistency, conflicts; process artifacts by year; inspector turnover. |
| `08_rater_reliability.ipynb` | Inter-rater reliability: same versus different inspector on consecutive inspections, stability of inspector effects, movers test, near-simultaneous inspections. |
| `09_citation_practice.ipynb` | What inspectors cite by tier, finding versus grading, life-hazard flagging, which checklist item is used for the same condition, hazards noted but not flagged. |
| `10_score_mechanics.ipynb` | Rubric arithmetic (3-unit minimum sample, zeroed units), sampling odds, scores near the cut-offs. |
| `11_external_validity.ipynb` | Raw and inspector-adjusted scores against HUD REAC, DCAD condition, CoStar, evictions, 311 and crime; whether strict inspectors are biased or better at finding problems. |

Shared code: `holdout.py` (base table, splits, models, metrics), `features.py` (feature
builders), `temporal.py` (temporal design), `inspectors.py` (inspector identity reconciled
across sources), `reliability.py` (two-way effects, consecutive pairs, clustered bootstrap),
`pull_311.py` (Dallas Open Data pull). Each notebook opens with its own findings. Outputs are cleared before commit,
so the findings sections carry the numbers. The data inputs and their sources are listed
in the repo README.

## Results

Ensemble PR-AUC (mean rank of the three models). "Clean" uses only information from
before the inspection window.

| Design | Base rate | 02 baseline | Best clean model | With data from during the window |
|---|---|---|---|---|
| Chelsea mirror | 0.16 | 0.27 | 0.33 | 0.34 |
| Tract-grouped CV | 0.20 | 0.31 | 0.36 | 0.38 |
| Random 70/30 | 0.20 | 0.40 | 0.47 | 0.46 |
| Temporal, pooled FY2023-24 and FY2024-25 | 0.11 | 0.18 | 0.23 | 0.25 with inspector leniency |

On the Chelsea mirror the clean model ranks 42% of the test area's bottom-10 properties
into its top 20% (random: 20%; tract poverty alone: 25%). In the temporal design, ranking
by the property's last score gives 0.17, so the model adds about 6 points over the rule
closest to the risk-based schedule Dallas already runs.

**What helped**

- **Owner track record**, built from training-area outcomes only: the bottom-10 rate of the
  owner's other inspected properties, grouped by DCAD owner name, owner mailing address and
  CoStar true owner.
- **Nearby outcomes**: the bottom-10 rate of the 10 nearest inspected properties.
- **DCAD ownership history, 2016–2021**: bottom-10 properties had sold more recently and more
  often, and their values grew faster.
- **Eviction filings per unit**: real signal, mostly overlapping the features above. Ranked
  alone, concurrent filings score 0.26 on the mirror.
- **Own inspection history**, in the temporal design. It helps most for properties with no
  earlier inspection; with history, the model and the last-score rule tie.

**What did not help**

- DCAD building systems (heat, AC, frame, foundation, remodel year).
- CoStar class and star rating.
- Tax delinquency, which is too rare.
- Apartment-premise crime.
- Resident 311 complaints, which ranked alone barely beat tract poverty.
- Regression on the continuous score, which does worse than classification.

## Why it plateaus near twice the base rate

A single inspection score barely persists, and the inspector moves it about as much as
the property's own record does (notebook 06):

- **Earlier scores barely predict the next one.** For 2,145 properties inspected in two or
  more fiscal years, the earlier mean score explains 1.8% of the variance in the held-out
  latest score. The visit's inspector and year effect explains 2.9%.
- **Inspectors differ widely.** Bottom-10 rates run from 0.5% at the 10th-percentile
  inspector to 7.5% at the median and 28% at the 90th. Inspectors work across the city, so
  this is not only geography (notebook 11 qualifies this).
- **Bottom-10 status repeats mainly with the same inspector.** It repeats 39% of the time
  when the same inspector returns, against 10% for other properties. With a different
  inspector it is 18% against 13%.
- **An inspector-adjusted target is no more predictable** from property features.
- **Inspector leniency is the strongest single lever.** As a process feature, the assigned
  inspector's earlier bottom-10 rate predicts about as well as the property's last score.
  It describes the inspection rather than the building.

## Inspection data quality (notebooks 07–11)

Is this garbage in, garbage out? Partly, and the problem is how the score is produced. The
records themselves are complete.

- **The records are complete.** Once Salesforce names and Code's backfill are reconciled,
  every inspection has an inspector. The sources agree on 97% of the inspections both name.
  The backfill's typed variants of the same person are mapped from the data (07).
- **The score reproduces within an inspector, not across them.**
  - On consecutive inspections in different fiscal years, scores correlate 0.41 with the same
    inspector and 0.055 with a different one. Bottom-10 agreement is kappa 0.21 against
    0.05.
  - Inspector effects have an SD of 4.9 points, about 11 points from strict to lenient. They
    replicate across halves of the properties (r 0.84), and a property's score change between
    inspectors tracks their difference one-for-one (movers slope 1.02) (08).
- **Practice differs.**
  - Inspectors write 1.6 to 11.9 citations per inspection, and 8 of 36 almost never cite a
    life-safety item. Citation volume accounts for about 45% of the inspector spread.
  - For AC, heat and hot water the checklist offers a specific item that zeroes the unit and a
    general one that almost never does. Inspectors pick the specific item 0% to 78% of the
    time, and the AC share fell from 67% to 20% over four years (09).
- **The rubric amplifies small things.** Up to 30 units only 3 units are sampled, so one
  zeroed unit costs 13.3 points, more than the inspector spread. 42% of inspections score
  exactly 100, and 29% of bottom-10 inspections sit within 2 points of the cut (10).
- **There is still real signal.**
  - Raw scores agree modestly with independent measures: DCAD condition 0.31, evictions 0.29
    (Spearman).
  - Removing inspector effects lowers that agreement, because strict inspectors work where
    distress is higher (r −0.57 with their properties' eviction rates) and partly find more
    (11).
  - An inspector's effect mixes leniency, assignment and detection.

So a model of bottom-10 status inherits all of this. Better inputs for a property-condition
model would be repeated inspections by different inspectors, the per-item severity points
the export lacks, and the interior unit sample itself.

## Comparison with Chelsea

Relative to its base rate, the Dallas model does better than Chelsea's. Chelsea's
published precision-recall curve, read off its figure, sits near 0.55–0.6 precision
against a 0.45 base rate, about 1.3×. Chelsea's code also re-fit its scaler on the test
set and scored average precision on 0/1 predictions instead of probabilities; neither is
carried over here.

## Cautions

- **Build from the July 2026 analytic files.** The repo's `properties_with_acs.csv`
  predates the June 2026 pipeline rebuild. These notebooks build from the July 2026
  analytic `.dta` files and take only tract ACS columns from the CSV.
- **Inspector names need reconciling.** `multitenant_inspections.dta` keeps only the first
  observation row's name and was saved before the pipeline's backfill. `inspectors.py`
  combines Salesforce names from every observation row with Code's July 2026 backfill. It
  maps typed variants to one person and flags the 2.6% of inspections where the sources
  name different people. Outputs show inspector codes. The data come from a public
  information request, so names aren't confidential, but the codes keep the work about
  practice until the team decides how to share it.
- **Leakage exclusions.** The owner priority and PE flags are excluded as leakage. The
  PE flag is set by name for owners identified from the distress results.
- **Data handling.** CoStar data is licensed and internal. The eviction table lives in a
  personal Databricks sandbox and is not governed. Only aggregate metrics appear in these
  notebooks.
- **Leak checks.** Shuffled-label checks in 03, 04 and 05 score at prevalence. All four
  notebooks were re-executed against the committed modules before commit.

## Open items

- A human review of methods and leakage reasoning, especially the fold-built owner and
  neighbor features, and the timing classes in notebook 04.
- Whether inspector assignment is known before a visit. That decides if inspector
  leniency can be used operationally. Code says inspectors are assigned by area and
  adjusted for staffing and enforcement needs.
- Before inspector-level results go to Code Compliance, agree the framing with the
  multitenant analysis lead and the housing lead. A large effect isn't necessarily bias
  (notebook 11).
- Ask Code Compliance for per-item severity points and the list of units sampled, which
  the export lacks. They would separate grading from sampling.
- The 311 pull sits in a local gitignored folder. If it should persist, it belongs in
  the Databricks sandbox.
