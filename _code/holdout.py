"""Shared evaluation harness for the Chelsea-style holdout notebooks (03 onward).

`02_spatial_holdout.ipynb` is the self-contained baseline record. This module
reproduces its base table, splits, models and metrics so follow-up notebooks
change one thing at a time against the same yardstick.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

DATA = Path(__file__).resolve().parent.parent / "_data"
SEED = 1
N_JOBS = 6  # more workers than this pushes an 8 GB machine into swap

ACS_COLS = [
    "pct_poverty", "pct_renter", "pct_hispanic", "pct_nh_black", "pct_nh_white",
    "pct_nh_asian", "pct_foreign_born", "pct_rent_burden_30", "pct_rent_burden_35",
]
RACE_COLS = ["pct_hispanic", "pct_nh_black", "pct_nh_white", "pct_nh_asian"]
PROPERTY_COLS = [
    "log_units", "units_per_building", "has_pool", "log_mkt_value", "log_value_per_unit",
    "imp_share", "sptd_multifamily", "out_of_state_owner", "log_owner_portfolio",
    "building_age", "log_bldg_area", "n_stories", "dcad_cond", "dcad_qual", "dcad_phys_depr_pct",
]
BASELINE_FEATURES = ACS_COLS + PROPERTY_COLS
TARGETS = ["bottom10", "bottom10_first"]
SPLITS = ["Chelsea mirror", "Random 70/30", "Grouped CV by tract"]

COND = {"POOR": 1, "FAIR": 2, "AVERAGE": 3, "GOOD": 4, "EXCELLENT": 5}
QUAL = {"LOW COST": 1, "AVERAGE": 2, "GOOD": 3, "VERY GOOD": 4, "EXCELLENT": 5}


# --------------------------------------------------------------------------- data

def _dcad_accounts(dcad: pd.DataFrame, accounts: set[str]) -> pd.DataFrame:
    """One row per DCAD account from commercial improvements (2021 if present, else 2026)."""
    has_2021 = set(dcad.loc[dcad.appraisal_year == 2021, "universal_id"])
    imp = dcad[(dcad.appraisal_year == 2021) | ~dcad.universal_id.isin(has_2021)]
    imp = imp[imp.universal_id.isin(accounts)].assign(
        cond=lambda d: d.property_condition.map(COND),
        qual=lambda d: d.property_quality.map(QUAL),
        year_built=lambda d: d.year_built.where(d.year_built.between(1850, 2026)),
        area=lambda d: d.gross_bldg_area.clip(lower=0),
    )
    imp["w"] = imp.area.where(imp.area > 0, 1)
    for c in ["cond", "qual", "phys_depr_pct"]:
        imp[f"{c}_wx"] = imp[c] * imp.w
        imp[f"{c}_w"] = imp.w.where(imp[c].notna())
    g = imp.groupby("universal_id")
    return pd.DataFrame({
        "year_built": g.year_built.min(),
        "bldg_area": g.area.sum(),
        "n_stories": g.num_stories.max(),
        **{
            f"dcad_{c}": g[f"{c}_wx"].sum(min_count=1) / g[f"{c}_w"].sum(min_count=1)
            for c in ["cond", "qual", "phys_depr_pct"]
        },
    }).reset_index()


def load_base(data: Path = DATA) -> pd.DataFrame:
    """The 02 notebook's analysis table: 2,796 properties, targets and baseline features."""
    props = pd.read_stata(data / "multitenant_properties.dta", convert_categoricals=False)
    insp = pd.read_stata(data / "multitenant_inspections.dta", convert_categoricals=False)
    acs = pd.read_csv(
        data / "properties_with_acs.csv",
        usecols=["address", "tract_geoid", *ACS_COLS],
        dtype={"tract_geoid": str},
    )
    parcel = pd.read_csv(data / "parcel_match.csv", usecols=["address", "universal_id"], dtype=str)
    dcad = pd.read_parquet(data / "dcad_improvement_commercial.parquet")
    dcad_acct = _dcad_accounts(dcad, set(parcel.universal_id.dropna()))

    insp = insp[insp.address.str.strip() != ""]
    bldg = insp.groupby("address").agg(n_buildings=("n_buildings", "max"), n_pools=("n_pools", "max"))
    first = (
        insp.dropna(subset=["score"])
        .sort_values(["address", "date_opened", "sr_number"])
        .groupby("address")
        .head(1)
        .set_index("address")["bottom10"]
        .rename("bottom10_first")
    )

    base = props[props.address.str.strip() != ""].copy()
    for c in ["owner_state", "sptd_code"]:
        base[c] = base[c].replace({"NA": np.nan, "": np.nan})

    df = (
        base.merge(acs, on="address", how="left")
        .merge(parcel, on="address", how="left")
        .merge(dcad_acct, on="universal_id", how="left")
        .merge(bldg, on="address", how="left")
        .merge(first, on="address", how="left")
    )
    df = df.dropna(subset=["pct_poverty"]).reset_index(drop=True)

    units = df.n_units.where(df.n_units > 0)
    mkt = df.total_market_value.where(df.total_market_value > 0)
    return df.assign(
        log_units=np.log1p(units),
        units_per_building=units / df.n_buildings.where(df.n_buildings > 0),
        has_pool=(df.n_pools > 0).astype(float).where(df.n_pools.notna()),
        log_mkt_value=np.log(mkt),
        log_value_per_unit=np.log(mkt / units),
        imp_share=df.improvement_value / mkt,
        sptd_multifamily=(df.sptd_code == "B11").astype(float).where(df.sptd_code.notna()),
        out_of_state_owner=(df.owner_state != "TEXAS").astype(float).where(df.owner_state.notna()),
        log_owner_portfolio=np.log(df.n_props_total.where(df.n_props_total > 0)),
        building_age=2021 - df.year_built,
        log_bldg_area=np.log(df.bldg_area.where(df.bldg_area > 0)),
    )


# --------------------------------------------------------------------------- evaluation

def poverty_cut(df: pd.DataFrame) -> float:
    return df.pct_poverty.quantile(2 / 3)


def splits(d: pd.DataFrame, y: np.ndarray, cut: float) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    idx = np.arange(len(d))
    area = d.pct_poverty.to_numpy() >= cut
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    groups = d.tract_geoid.fillna("missing").to_numpy()
    return {
        "Chelsea mirror": [(idx[area], idx[~area])],
        "Random 70/30": [(tr, te)],
        "Grouped CV by tract": list(GroupKFold(n_splits=5).split(idx, y, groups)),
    }


def pipe(est, scale: bool = False):
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, est)


MODELS = {
    "LASSO": (
        pipe(LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=5000, random_state=SEED), scale=True),
        {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
    ),
    "Random forest": (
        pipe(RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=1)),
        {
            "randomforestclassifier__max_depth": [6, None],
            "randomforestclassifier__min_samples_leaf": [5, 20],
        },
    ),
    "XGBoost": (
        pipe(XGBClassifier(
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, eval_metric="logloss",
        )),
        {
            "xgbclassifier__max_depth": [2, 4],
            "xgbclassifier__learning_rate": [0.05],
            "xgbclassifier__min_child_weight": [1, 5],
        },
    ),
}


# Regressors fit a continuous score (negated, so higher = riskier) and rank by prediction.
REGRESSORS = {
    "Ridge (score)": (
        pipe(Ridge(), scale=True),
        {"ridge__alpha": [1.0, 10.0, 100.0, 1000.0]},
    ),
    "Random forest (score)": (
        pipe(RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=1)),
        {
            "randomforestregressor__max_depth": [6, None],
            "randomforestregressor__min_samples_leaf": [5, 20],
        },
    ),
    "XGBoost (score)": (
        pipe(XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1)),
        {
            "xgbregressor__max_depth": [2, 4],
            "xgbregressor__learning_rate": [0.05],
            "xgbregressor__min_child_weight": [1, 5],
        },
    ),
}


def fit(name: str, X: pd.DataFrame, y: np.ndarray, models: dict = MODELS):
    est, grid = models[name]
    if name in REGRESSORS:
        cv, scoring = KFold(n_splits=5, shuffle=True, random_state=SEED), "neg_mean_squared_error"
    else:
        cv, scoring = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), "average_precision"
    gs = GridSearchCV(clone(est), grid, scoring=scoring, cv=cv, n_jobs=N_JOBS)
    return gs.fit(X, y).best_estimator_


def predict(est, X: pd.DataFrame) -> np.ndarray:
    """Risk score: class probability for classifiers, the prediction itself for regressors."""
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.predict(X)


def found_at(y: np.ndarray, s: np.ndarray, k: float) -> float:
    """Share of all positives that fall in the top k of the ranking (random tie-break)."""
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(len(s)), -s))
    return y[order[: int(np.ceil(k * len(y)))]].sum() / y.sum()


def scores(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    return {
        "PR-AUC": average_precision_score(y, s),
        "ROC-AUC": roc_auc_score(y, s),
        **{f"found@{int(k * 100)}%": found_at(y, s, k) for k in (0.1, 0.2, 0.3)},
    }


# Builds extra feature columns for every row of `d` using labels from the training rows
# only (e.g. owner track records). Called once per fold.
FoldFeatures = Callable[[pd.DataFrame, np.ndarray, str], pd.DataFrame]


def run(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    models: dict = MODELS,
    which: set[str] | None = None,
    fold_features: FoldFeatures | None = None,
    baselines: dict[str, str] | None = None,
    label: str = "",
    regress_on: str | None = None,
) -> pd.DataFrame:
    """Score each model on each split; out-of-fold predictions pooled for grouped CV.

    `baselines` maps a ranker name to a single column that ranks properties on its own
    (higher = riskier); "Random" and "Poverty only" are always included. With
    `regress_on`, models fit that continuous column (negated: lower score = riskier) and are
    still scored on `target`; pass `models=REGRESSORS`.
    """
    d = df.dropna(subset=[target]).reset_index(drop=True)
    y = d[target].astype(int).to_numpy()
    y_fit = -d[regress_on].to_numpy(dtype=float) if regress_on else y
    cut = poverty_cut(df)
    baselines = {"Poverty only": "pct_poverty", **(baselines or {})}
    rows = []
    for split_name, folds in splits(d, y, cut).items():
        if which and split_name not in which:
            continue
        test = np.concatenate([te for _, te in folds])
        yt = y[test]
        base = {"label": label, "target": target, "split": split_name,
                "n_train": len(folds[0][0]), "n_test": len(test), "prevalence_test": yt.mean()}
        rows.append({**base, "ranker": "Random", "PR-AUC": yt.mean(), "ROC-AUC": 0.5,
                     "found@10%": 0.1, "found@20%": 0.2, "found@30%": 0.3})
        for name, col in baselines.items():
            s = d[col].fillna(d[col].median()).to_numpy()[test]
            rows.append({**base, "ranker": name, **scores(yt, s)})
        preds = {m: np.empty(len(d)) for m in models}
        for tr, te in folds:
            X = d[features]
            if fold_features is not None:
                X = pd.concat([X, fold_features(d, tr, target)], axis=1)
            for m in models:
                fit_rows = tr[np.isfinite(y_fit[tr])]  # regression targets can be missing
                preds[m][te] = predict(fit(m, X.iloc[fit_rows], y_fit[fit_rows], models), X.iloc[te])
        for m in models:
            rows.append({**base, "ranker": m, **scores(yt, preds[m][test])})
        if len(models) > 1:
            # mean percentile rank across models, computed on the test rows
            ens = np.mean([pd.Series(preds[m][test]).rank(pct=True) for m in models], axis=0)
            rows.append({**base, "ranker": "Ensemble", **scores(yt, ens)})
    return pd.DataFrame(rows)


def fit_split(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    split: str,
    model: str,
    *,
    fold_features: FoldFeatures | None = None,
    models: dict = MODELS,
):
    """Fit one model on one single-fold split; return (estimator, X_test, y_test)."""
    d = df.dropna(subset=[target]).reset_index(drop=True)
    y = d[target].astype(int).to_numpy()
    ((tr, te),) = splits(d, y, poverty_cut(df))[split]
    X = d[features]
    if fold_features is not None:
        X = pd.concat([X, fold_features(d, tr, target)], axis=1)
    return fit(model, X.iloc[tr], y[tr], models), X.iloc[te], y[te]


def compare(results: pd.DataFrame, metric: str = "PR-AUC") -> pd.DataFrame:
    """Wide view: one row per label x ranker, one column per split, `metric` values."""
    return results.pivot_table(index=["label", "ranker"], columns="split", values=metric, sort=False)[
        [s for s in SPLITS if s in set(results.split)]
    ]
