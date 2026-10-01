# Multitenant Inspection Prediction

A first-pass at predicting **which Dallas multitenant properties end up
distressed**, using City of Dallas Code Compliance graded-inspection data. It
replicates the modeling approach from the Chelsea code-violations study (train a
few models on property + neighborhood features, see what predicts a bad outcome,
look at which features matter).

This is the repo for any  **prediction / machine-learning** multitenant inspection work.

Private repository under the CPAL organization.

---

## What's in here, and what's not

Code and documentation only. **No data, no credentials.** The inspection records
include property addresses and owner names, so data files never get committed — the
`.gitignore` enforces this.

Commit these:
- `_code/` — the analysis notebook(s)
- `_docs/` — the data dictionary, methodology notes, and the GitHub workflow guide
- `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md`, `SETUP.md`

Never commit these:
- `_data/` — the input CSV (has addresses + owner names)
- `_output/` — plots and any exported tables

The data lives on the shared drive at:
```
G:\Shared drives\CPAL Data Team\_Housing\_Multitenant_Inspection\_output\properties_with_acs.csv
```
Copy that one file into your local `_data/` folder. It stays on your machine and is
never pushed.

---

## Quickstart

Read **[SETUP.md](SETUP.md)** — it walks through everything
from scratch. The short version:

```bash
uv sync                       # create the environment and install all dependencies
# copy properties_with_acs.csv into _data/  (from the shared drive, see above)
uv run jupyter lab            # launch JupyterLab, then open _code/01_replicate_bottom10.ipynb
```

---

## The exercise

We are predicting the column **`bottom10`** — whether a property has ever scored in
the bottom 10% of inspection scores (the closest analog to Chelsea's "any violation"
target). The notebook [_code/01_replicate_bottom10.ipynb](_code/01_replicate_bottom10.ipynb)
is a skeleton with section headers and TODOs; you fill in the code.

Before writing any modeling code, read:
- **[_docs/data_dictionary.md](_docs/data_dictionary.md)** — every column explained, and
  **which columns you must NOT use as features** (several are derived from the score and
  would leak the answer).
- **[_docs/chelsea_methodology.md](_docs/chelsea_methodology.md)** — what the original
  study did, and a place to add your own Chelsea notes.

---

## First tasks (each one = one branch → one pull request)

This is also how you'll learn the GitHub pull-request workflow. See
**[_docs/github_workflow.md](_docs/github_workflow.md)** for the exact git commands.
Do these in order; keep each PR small.

1. **Environment + data load.** Run `uv sync`, copy the CSV into `_data/`, open the
   notebook, run the load cell, and confirm it prints `(2798, 33)`. (A tiny first PR —
   e.g. add your name to a contributors line — just to practice the flow.)
2. **Add your Chelsea notes** to `_docs/chelsea_methodology.md` under the "Your notes"
   heading.
3. **EDA.** What share of properties are `bottom10`? Plot distributions of the candidate
   features; check for missing values.
4. **Build the feature matrix.** Select the *usable* features from the data dictionary
   (not the leakage columns), and handle any missing values.
5. **Train/test split.** A random split, stratified on `bottom10`.
6. **Baseline model.** Logistic regression / LASSO.
7. **Tree models.** RandomForest and XGBoost.
8. **Evaluation.** Precision-recall AUC, ROC-AUC, and a confusion matrix.
9. **Feature importances.** Plot them and save the figure to `_output/`.
10. **Write-up.** A few paragraphs comparing your results to Chelsea's, and noting the
    big caveat (see the data dictionary): Dallas inspects *every* multitenant property,
    so this model describes *which property/neighborhood profiles are associated with
    distress* — not *where to inspect next*, which is what Chelsea predicted.
- Dhatri completed Issue #1 🎉

---

## Holdout test and follow-up notebooks (02–06)

These test the Chelsea design on Dallas, then try to improve on it. Each notebook opens
with its findings. Outputs are cleared before commit, so the findings carry the numbers.

| Notebook | What it does |
|---|---|
| `02_spatial_holdout.ipynb` | Baseline. Train on the highest-poverty third of tracts and score the rest (the Chelsea mirror), with a random split and tract-grouped CV for reference. Self-contained. |
| `03_engineered_features.ipynb` | DCAD building systems, spatial context, owner track record, nearby outcomes. |
| `04_supplemental_sources.ipynb` | Eviction filings, DCAD roll history, DCTO tax roll, CoStar, DPD apartment-premise crime, Dallas 311. Each tagged pre-window, during or snapshot. |
| `05_temporal.ipynb` | Rolling-origin design: predict each property's first inspection of a fiscal year from what was known on 1 October. |
| `06_outcome_noise.ipynb` | How much of a score is the inspector; inspector-adjusted target; regression against classification. |

Notebooks 03–06 share `_code/holdout.py` (base table, splits, models, metrics),
`_code/features.py` (feature builders) and `_code/temporal.py` (the temporal design).

**Inputs in `_data/`** (all gitignored):

| File | Source |
|---|---|
| `multitenant_properties.dta`, `multitenant_inspections.dta` | shared drive `_Multitenant_Inspection/_data/_analytic/` |
| `properties_with_acs.csv` | shared drive `_Multitenant_Inspection/_output/` (tract ACS columns only) |
| `parcel_match.csv` | shared drive `_Multitenant_Inspection/_data/_processed/` |
| `code_inspector_backfill_2026.xlsx` | shared drive `_Multitenant_Inspection/_data/_raw/_datafromcode2026/CPAL - MISSING INSPECTOR DATA 2021-2025.xlsx` |
| `dcad_improvement_commercial.parquet`, `dcad_improvement_commercial_all_years.parquet` | `silver.silver_tx_dallas_cad_car.improvement_commercial` (query in notebook 02) |
| `dcad_account.parquet`, `dcad_deed.parquet`, `dcad_owner.parquet` | `dev_silver.silver_tx_dallas_cad_car.account`, `deed`, `account_owner`, filtered to the inspected accounts |
| `tax_summary.parquet` | `silver.silver_tx_dallas_tax.tax_account_summary` |
| `evictions_parcel_year.parquet`, `evictions_parcel_month.parquet` | `dev_sandbox.taylor.evictions_north_tx_landlord`, Dallas County, distinct cases per parcel per year or month |
| `costar_mf.parquet` | `gold.housing.costar_03312026`, Dallas County multifamily (licensed; internal only) |
| `nibrs_apartment.parquet` | `gold.safety.nibrs_crime`, apartment premises, one row per incident |
| `dcad_parcel_centroids_2025.parquet` | 2025 centroids of `dev_bronze.bronze_tx_dallas_cad.parcel_geom_certified` |
| `311_code_compliance.parquet` | Dallas Open Data `4n3h-rfqm`, pulled with `uv run python _code/pull_311.py` |

Run a notebook with `uv run jupyter nbconvert --to notebook --execute --inplace _code/<name>.ipynb`.
Notebook 04 takes about 15 minutes. Parallel model fitting is capped at six workers
(`holdout.N_JOBS`) to keep an 8 GB machine out of swap.

