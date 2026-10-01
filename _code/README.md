# Notebooks: testing the Chelsea inspection model on Dallas

> **Status: exploratory and unreviewed.** Notebooks 03–06 and this summary come from an
> autonomous exploration by Claude Opus 5.5 (Anthropic) on 2026-10-01, run at the repo
> owner's request without human supervision. No person has checked the feature choices,
> methods or numbers. Treat the results as leads to verify. Notebook 02 was also written by
> Claude, from a plan the repo owner reviewed and approved before it ran.

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

Shared code: `holdout.py` (base table, splits, models, metrics), `features.py` (feature
builders), `temporal.py` (temporal design, inspector backfill), `pull_311.py` (Dallas Open
Data pull). Each notebook opens with its own findings. Outputs are cleared before commit,
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
| Temporal, pooled FY2023-24 and FY2024-25 | 0.11 | 0.18 | 0.23 | 0.26 with inspector leniency |

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
  latest score. The visit's inspector and year effect explains 3.1%.
- **Inspectors differ widely.** Bottom-10 rates run from 0.8% at the 10th-percentile
  inspector to 8% at the median and 32% at the 90th. Inspectors work across the city, so
  this is not geography.
- **Bottom-10 status repeats mainly with the same inspector.** It repeats 39% of the time
  when the same inspector returns, against 10% for other properties. With a different
  inspector it is 18% against 13%.
- **An inspector-adjusted target is no more predictable** from property features.
- **Inspector leniency is the strongest single lever.** As a process feature, the assigned
  inspector's earlier bottom-10 rate predicts about as well as the property's last score.
  It describes the inspection rather than the building.

Relative to its base rate, the Dallas model does better than Chelsea's. Chelsea's
published precision-recall curve, read off its figure, sits near 0.55–0.6 precision
against a 0.45 base rate, about 1.3×. Chelsea's code also re-fit its scaler on the test
set and scored average precision on 0/1 predictions instead of probabilities; neither is
carried over here.

## Cautions

- **Build from the July 2026 analytic files.** The repo's `properties_with_acs.csv`
  predates the June 2026 pipeline rebuild. These notebooks build from the July 2026
  analytic `.dta` files and take only tract ACS columns from the CSV.
- **Inspector names need the backfill.** `multitenant_inspections.dta` was saved before the
  pipeline's inspector backfill, so names are re-filled from Code Compliance's July 2026
  delivery. They are replaced with anonymous codes on load and appear in no output.
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
  leniency can be used operationally.
- The 311 pull sits in a local gitignored folder. If it should persist, it belongs in
  the Databricks sandbox.
