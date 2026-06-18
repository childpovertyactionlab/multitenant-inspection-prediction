# Data dictionary — `properties_with_acs.csv`

One row per **property** (unique address). 2,798 rows, 33 columns. Produced by Abel's
Stata + R pipeline; this is the cleaned, property-level output, so you can skip all the
data wrangling and go straight to modeling.

> **Read this whole file before you build features.** Several columns are derived from
> the inspection *score*, which is also what the target is derived from. Using them as
> predictors would "leak" the answer and give you a model that looks amazing and means
> nothing. The table below marks each column **TARGET**, **FEATURE (usable)**, or
> **LEAK — do not use as a feature**.

---

## The target

| Column | Role | Notes |
|---|---|---|
| `bottom10` | **TARGET** | 1 if the property was ever in the bottom 10% of inspection scores (score ≤ 86), else 0. ~19.7% are positive. This is what you predict. |

---

## Identifiers (not features — drop before modeling)

| Column | Role | Notes |
|---|---|---|
| `address` | ID | Property street address. PII — never goes in a committed file. |
| `account_name` | ID | DCAD account / owner-facing name. |
| `sr_owner` | ID | Owner field from the service request. |
| `tract_geoid` | ID | Census tract GEOID. Could be used as a high-cardinality categorical, but treat as ID for the first pass. |
| `tract_name` | ID | Human-readable tract name. |

---

## LEAK — do NOT use as features

All of these are computed *from the inspection scores or findings*, i.e. from the same
thing `bottom10` comes from. A model using them is circular.

| Column | Why it leaks |
|---|---|
| `score_current` | The score itself. `bottom10` is literally derived from scores. |
| `score_min` | Min score → directly determines `bottom10`. |
| `score_max` | Score-derived. |
| `score_mean` | Score-derived. |
| `score_range` | `score_max − score_min`, score-derived. |
| `n_bottom10` | Count of bottom-10 inspections — defines the target. |
| `chronic` | Bottom-10 in 2+ fiscal years — a stricter version of the target. |
| `n_inspections` | Borderline; correlated with how distressed/re-visited a property is. Leave out of the first pass to be safe. |
| `life_hazard` | An inspection *finding*, not a property trait known beforehand. |
| `flag_admin_failure` | Inspection finding. |
| `flag_co_missing` | Inspection finding. |
| `flag_mt_cert_missing` | Inspection finding. |
| `flag_crime_prev_missing` | Inspection finding. |
| `flag_master_meter_missing` | Inspection finding. |
| `flag_mpo_missing` | Inspection finding. |
| `flag_pool_permit_missing` | Inspection finding. |

---

## FEATURE (usable) — independent of the inspection result

These describe the property's *neighborhood* (and its location), known without doing an
inspection. They're your starting feature set.

| Column | Role | Notes |
|---|---|---|
| `pct_poverty` | FEATURE | Tract poverty rate. |
| `pct_renter` | FEATURE | Tract renter share. |
| `pct_hispanic` | FEATURE | Tract Hispanic share. |
| `pct_nh_black` | FEATURE | Tract non-Hispanic Black share. |
| `pct_nh_white` | FEATURE | Tract non-Hispanic White share. |
| `pct_nh_asian` | FEATURE | Tract non-Hispanic Asian share. |
| `pct_foreign_born` | FEATURE | Tract foreign-born share. |
| `pct_rent_burden_30` | FEATURE | Share of renters paying ≥30% of income on rent. |
| `pct_rent_burden_35` | FEATURE | Share paying ≥35%. |
| `lon` | FEATURE | Longitude. Usable as a raw geographic feature. |
| `lat` | FEATURE | Latitude. |

### Want a richer feature set?
The neighborhood features above are all tract-level. To add a real *property-level*
trait, merge `n_units` (number of housing units) from `property_profiles.csv` on
`address`:
```python
profiles = pd.read_csv("_data/property_profiles.csv")
df = df.merge(profiles[["address", "n_units"]], on="address", how="left")
```
(`n_units` is a property characteristic, not an inspection finding, so it's a safe
feature.) This is a good optional pandas exercise — but get the basic model working first.

---

## The big methodology caveat (put this in your write-up)

The Chelsea study trained on the *subset of properties that had been inspected* and then
predicted hazards for the rest of the city that **had not** been inspected — so the model
was a tool for *deciding where to inspect next*.

**Dallas is different: every multitenant property gets inspected** under the graded
program. So there is no un-inspected set to predict for. Your train/test split is just a
random split for evaluation, and the model answers a different question:

> *Which property / neighborhood profiles are associated with ending up distressed?*

That's still useful (it surfaces patterns), but it is **not** "where should we inspect
next." Be explicit about this difference when you compare your results to Chelsea's.
