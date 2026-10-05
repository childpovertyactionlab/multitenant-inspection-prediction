"""Temporal (rolling-origin) design: predict each property's first inspection of a fiscal
year from what was known at the start of that year.

Unit: property × fiscal year, the first scored inspection in that year. Every history
feature comes from strictly earlier fiscal years, so within-year follow-up visits after
a bad score never inform the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import holdout as h
from features import EARTH_KM, sponsor_key

FY_ORDER = ["FY2020-21", "FY2021-22", "FY2022-23", "FY2023-24", "FY2024-25"]
FLAGS = [
    "life_hazard", "flag_admin_failure", "flag_co_missing", "flag_crime_prev_missing",
    "flag_master_meter_missing", "flag_mpo_missing", "flag_mt_cert_missing",
    "flag_pool_permit_missing",
]
HISTORY_COLS = [
    "has_history", "n_prior", "last_score", "min_score", "mean_score", "n_prior_b10",
    "any_prior_b10", "last_b10", "score_trend", "years_since_last", "n_prior_flags",
    *[f"prior_{c}" for c in FLAGS],
]
CONTEXT_COLS = ["sponsor_prior_b10_rate", "sponsor_prior_n", "knn_prior_b10_rate", "knn_prior_km"]
ACTIVITY_COLS = ["ev_prior_12m", "ev_prior_36m", "c311_prior_12m", "c311_prior_36m"]
FOLDS = [
    (["FY2021-22", "FY2022-23"], "FY2023-24"),
    (["FY2021-22", "FY2022-23", "FY2023-24"], "FY2024-25"),
]


def load_inspections(data=h.DATA) -> pd.DataFrame:
    """Scored inspections with the reconciled inspector (see `inspectors.py`).

    `inspector_id` is a stable code (I01, ...); `inspector` is an integer version for
    models (-1 when unknown). The analytic file's own `assessor_name` is dropped: it kept only
    the first observation row's name per inspection and predates Code's backfill.
    """
    from inspectors import inspector_table

    insp = pd.read_stata(data / "multitenant_inspections.dta", convert_categoricals=False)
    insp["sr_number"] = insp.sr_number.astype(str).str.strip()
    who = inspector_table(data)[["sr_number", "inspector_id", "source", "conflict"]]
    insp = insp.drop(columns="assessor_name").merge(
        who.rename(columns={"source": "inspector_source", "conflict": "inspector_conflict"}),
        on="sr_number", how="left",
    )
    insp["inspector"] = pd.factorize(insp.inspector_id, use_na_sentinel=True)[0]
    insp = insp[(insp.address.str.strip() != "") & insp.score.notna()].copy()
    insp["fy"] = insp.fiscal_year.map({fy: i for i, fy in enumerate(FY_ORDER)})
    return insp.sort_values(["address", "date_opened", "sr_number"]).reset_index(drop=True)


def build_units(insp: pd.DataFrame, base: pd.DataFrame,
                owner_keys: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Property × FY units with own-history, owner and neighbour features as of FY start."""
    units = insp.groupby(["address", "fy"]).head(1)[
        ["address", "fy", "fiscal_year", "date_opened", "score", "bottom10", "inspector"]
    ].rename(columns={"score": "unit_score", "bottom10": "target"})
    units = units[units.fy >= 1].reset_index(drop=True)  # FY2020-21 is history only

    rows = []
    for fy in sorted(units.fy.unique()):
        prior = insp[insp.fy < fy]
        g = prior.groupby("address")
        last = g.tail(1).set_index("address")
        firsts = g.head(1).set_index("address")
        hist = pd.DataFrame({
            "n_prior": g.size(),
            "min_score": g.score.min(),
            "mean_score": g.score.mean(),
            "n_prior_b10": g.bottom10.sum(),
            "last_score": last.score,
            "last_b10": last.bottom10,
            "score_trend": last.score - firsts.score,
            "last_date": last.date_opened,
            **{f"prior_{c}": g[c].max() for c in FLAGS},
        })
        hist["n_prior_flags"] = hist[[f"prior_{c}" for c in FLAGS]].sum(axis=1)
        u = units[units.fy == fy].merge(hist, left_on="address", right_index=True, how="left")
        u["years_since_last"] = (u.date_opened - u.last_date).dt.days / 365.25
        rows.append(u.drop(columns="last_date"))
    units = pd.concat(rows, ignore_index=True)
    units["has_history"] = units.n_prior.notna().astype(float)
    units["any_prior_b10"] = (units.n_prior_b10 > 0).astype(float).where(units.n_prior.notna())
    units["n_prior"] = units.n_prior.fillna(0)

    units = units.merge(base, on="address", how="inner", suffixes=("", "_prop"))
    return add_prior_context(units, insp, base, owner_keys)


def add_prior_context(units: pd.DataFrame, insp: pd.DataFrame, base: pd.DataFrame,
                      owner_keys: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Owner groups' and neighbours' bottom-10 record in earlier years (other properties
    only). `owner_keys` maps a feature prefix to a grouping key aligned with `base`; the
    mailing-address sponsor key is always included."""
    keys = {"sponsor": sponsor_key(base), **(owner_keys or {})}
    keys = {k: pd.Series(v.to_numpy(), index=base.address.to_numpy()) for k, v in keys.items()}
    coords = base.set_index("address")[["latitude", "longitude"]]
    out = []
    for fy in sorted(units.fy.unique()):
        prior = insp[insp.fy < fy]
        per_prop = prior.groupby("address").bottom10.max().rename("ever_b10").to_frame()
        u = units[units.fy == fy].copy()
        prior_rate = per_prop.ever_b10.mean()
        # remove the property's own record from its group's tally
        own = per_prop.ever_b10.reindex(u.address).fillna(0).to_numpy()
        own_seen = u.address.isin(per_prop.index).to_numpy().astype(float)
        for name, key in keys.items():
            grp = per_prop.assign(k=key.reindex(per_prop.index).to_numpy()).dropna(subset=["k"])
            s = grp.groupby("k").ever_b10.agg(["sum", "count"])
            k_u = pd.Series(key.reindex(u.address).to_numpy())
            has = k_u.notna().to_numpy()
            g_sum = k_u.map(s["sum"]).fillna(0).to_numpy() - own * own_seen * has
            g_n = k_u.map(s["count"]).fillna(0).to_numpy() - own_seen * has
            u[f"{name}_prior_n"] = g_n
            u[f"{name}_prior_b10_rate"] = (g_sum + 5 * prior_rate) / (g_n + 5)

        ref = coords.reindex(per_prop.index).dropna()
        tree = BallTree(np.radians(ref.to_numpy(dtype=float)), metric="haversine")
        q = np.radians(coords.reindex(u.address).to_numpy(dtype=float))
        dist, idx = tree.query(q, k=11)
        is_self = ref.index.to_numpy()[idx] == u.address.to_numpy()[:, None]
        order = np.argsort(is_self, axis=1, kind="stable")[:, :10]
        labels = per_prop.ever_b10.reindex(ref.index).to_numpy()[np.take_along_axis(idx, order, axis=1)]
        u["knn_prior_b10_rate"] = labels.mean(axis=1)
        u["knn_prior_km"] = np.take_along_axis(dist, order, axis=1).mean(axis=1) * EARTH_KM
        out.append(u)
    return pd.concat(out, ignore_index=True)


def run(units: pd.DataFrame, features: list[str], *, models: dict = h.MODELS, label: str = "",
        baselines: dict[str, pd.Series] | None = None, regress_on: str | None = None) -> pd.DataFrame:
    """Rolling-origin evaluation. Reports each test year and the two years pooled.
    With `regress_on`, models fit that continuous column negated (pass `models=h.REGRESSORS`)."""
    y = units.target.astype(int).to_numpy()
    y_fit = -units[regress_on].to_numpy(dtype=float) if regress_on else y
    baselines = baselines or {}
    preds = {m: np.full(len(units), np.nan) for m in models}
    test_all = []
    for train_fys, test_fy in FOLDS:
        tr = np.flatnonzero(units.fiscal_year.isin(train_fys).to_numpy())
        te = np.flatnonzero((units.fiscal_year == test_fy).to_numpy())
        test_all.append(te)
        X = units[features]
        for m in models:
            fit_rows = tr[np.isfinite(y_fit[tr])]
            preds[m][te] = h.predict(h.fit(m, X.iloc[fit_rows], y_fit[fit_rows], models), X.iloc[te])

    rows = []
    for name, te in [*zip([fy for _, fy in FOLDS], test_all), ("Pooled", np.concatenate(test_all))]:
        yt = y[te]
        base = {"label": label, "test": name, "n_test": len(te), "prevalence_test": yt.mean()}
        rows.append({**base, "ranker": "Random", "PR-AUC": yt.mean(), "ROC-AUC": 0.5,
                     "found@10%": 0.1, "found@20%": 0.2, "found@30%": 0.3})
        for bname, s in baselines.items():
            rows.append({**base, "ranker": bname, **h.scores(yt, s.to_numpy()[te])})
        for m in models:
            rows.append({**base, "ranker": m, **h.scores(yt, preds[m][te])})
        if len(models) > 1:
            # rank within each test year so pooled years are on one scale
            ranks = [
                pd.Series(preds[m][te]).groupby(units.fiscal_year.to_numpy()[te]).rank(pct=True)
                for m in models
            ]
            rows.append({**base, "ranker": "Ensemble", **h.scores(yt, np.mean(ranks, axis=0))})
    return pd.DataFrame(rows)


def fy_start(fy: int) -> pd.Timestamp:
    """First day of fiscal year index `fy` (0 = FY2020-21, which starts 2020-10-01)."""
    return pd.Timestamp(f"{2020 + fy}-10-01")


def add_prior_activity(units: pd.DataFrame, base: pd.DataFrame, parcel_sets: list[list[str]],
                       data=h.DATA, radius_m: float = 100.0) -> pd.DataFrame:
    """Eviction filings and resident 311 requests in the 12 and 36 months before each
    fiscal year starts, per 100 units per year."""
    ev = pd.read_parquet(data / "evictions_parcel_month.parquet")
    month = pd.to_datetime(ev.month)
    ev["month"] = month.dt.tz_convert(None) if month.dt.tz is not None else month  # Databricks returns UTC
    r = pd.read_parquet(data / "311_code_compliance.parquet",
                        columns=["service_request_number", "method_received_description", "created_date",
                                 "latitude", "longitude"])
    insp_sr = set(pd.read_stata(data / "multitenant_inspections.dta", convert_categoricals=False).sr_number.astype(str))
    r = r[~r.method_received_description.isin(["Proactive", "Internal"]) & ~r.service_request_number.isin(insp_sr)]
    sets = pd.Series(parcel_sets, index=base.address.to_numpy())
    units_by_addr = base.set_index("address").n_units
    xy = base.set_index("address")[["latitude", "longitude"]]

    out = []
    for fy in sorted(units.fy.unique()):
        start = fy_start(fy)
        u = units[units.fy == fy].copy()
        n_units = units_by_addr.reindex(u.address).where(lambda s: s > 0).to_numpy()
        for months in (12, 36):
            a = start - pd.DateOffset(months=months)
            years = months / 12
            by_parcel = ev[(ev.month >= a) & (ev.month < start)].groupby("parcel_id").n_cases.sum()
            cnt = np.array([by_parcel.reindex(s).sum() if s else np.nan for s in sets.reindex(u.address)])
            u[f"ev_prior_{months}m"] = np.log1p(100 * cnt / n_units / years)
            pts = r[(r.created_date >= a) & (r.created_date < start)].dropna(subset=["latitude", "longitude"])
            tree = BallTree(np.radians(pts[["latitude", "longitude"]].to_numpy(dtype=float)), metric="haversine")
            c = tree.query_radius(np.radians(xy.reindex(u.address).to_numpy(dtype=float)),
                                  r=radius_m / 1000 / EARTH_KM, count_only=True)
            u[f"c311_prior_{months}m"] = np.log1p(100 * c / n_units / years)
        out.append(u)
    return pd.concat(out, ignore_index=True)


INSPECTOR_COLS = ["inspector_prior_b10_rate", "inspector_prior_n"]


def add_inspector_leniency(units: pd.DataFrame, insp: pd.DataFrame, a: float = 20.0) -> pd.DataFrame:
    """The unit's inspector's bottom-10 rate on other inspections in earlier fiscal years,
    smoothed toward the overall rate. A process feature: it describes who inspects, not the
    property, and is usable only where assignments are known before the visit."""
    out = []
    for fy in sorted(units.fy.unique()):
        prior = insp[(insp.fy < fy) & (insp.inspector >= 0)]
        s = prior.groupby("inspector").bottom10.agg(["sum", "count"])
        rate = prior.bottom10.mean()
        u = units[units.fy == fy].copy()
        n = u.inspector.map(s["count"]).fillna(0)
        u["inspector_prior_n"] = n
        u["inspector_prior_b10_rate"] = (u.inspector.map(s["sum"]).fillna(0) + a * rate) / (n + a)
        out.append(u)
    return pd.concat(out, ignore_index=True)
