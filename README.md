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
