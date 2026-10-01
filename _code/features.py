"""Feature builders for the follow-up notebooks.

Two kinds:
- static builders take the base table and return new columns known without inspecting
  the property;
- fold builders (`FoldFeatures`) use labels from the training rows only and are rebuilt
  inside every split, so test-area outcomes never reach the features.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

from holdout import DATA, SEED

EARTH_KM = 6371.0
DALLAS_CBD = (32.7767, -96.7970)


# --------------------------------------------------------------------------- DCAD detail

def dcad_detail(df: pd.DataFrame, data: Path = DATA) -> pd.DataFrame:
    """Building systems and construction from DCAD commercial improvements, 2021 snapshot.

    Area-weighted shares across an account's buildings. Accounts missing from 2021 use their
    earliest later year. The table's value columns are entirely null in the platform silver,
    so no value features come from here.
    """
    imp = pd.read_parquet(data / "dcad_improvement_commercial_all_years.parquet")
    imp = imp[imp.universal_id.isin(set(df.universal_id.dropna()))]
    first_year = imp.groupby("universal_id").appraisal_year.transform("min").clip(lower=2021)
    snap = imp[imp.appraisal_year == first_year].copy()
    snap["w"] = snap.gross_bldg_area.clip(lower=0).where(lambda a: a > 0, 1)
    apt = snap.bldg_class_desc.str.startswith("APARTMENT", na=False)

    flags = {
        "central_heat": snap.heating_type.isin(["CENTRAL HEAT", "CENTRAL PARTIAL"]),
        "no_heat_system": snap.heating_type.isin(["NONE", "HEATERS", "GAS HEATERS", "FLOOR FURNACE", "WALL"]),
        "central_ac": snap.ac_type.isin(["CENTRAL A/C", "CENTRAL PARTIAL", "CHILLED WATER", "FORCED AIR"]),
        "window_ac": snap.ac_type.isin(["WINDOW", "WALL"]),
        "wood_frame": snap.construction_type.eq("D-WOOD FRAME"),
        "pier_foundation": snap.foundation_type.isin(["PIERS", "BLOCKS", "POST"]),
        "frame_exterior": snap.bldg_class_desc.eq("APARTMENT (FRAME EXTERIOR)"),
    }
    # weight only apartment buildings for systems (garages, clubhouses have none)
    w_apt = snap.w.where(apt, 0)
    for k, v in flags.items():
        snap[f"{k}_wx"] = v.astype(float) * w_apt
    snap["w_apt"] = w_apt
    snap["remodel_year"] = snap.remodel_year.where(snap.remodel_year.between(1900, 2026))
    snap["funct_x"] = snap.funct_depr_pct.fillna(0) * snap.w
    snap["extrnl_x"] = snap.extrnl_depr_pct.fillna(0) * snap.w

    g = snap.groupby("universal_id")
    out = pd.DataFrame({
        **{f"share_{k}": g[f"{k}_wx"].sum() / g.w_apt.sum().replace(0, np.nan) for k in flags},
        "n_dcad_buildings": g.size(),
        "years_since_remodel": 2021 - g.remodel_year.max(),
        "dcad_funct_depr": g.funct_x.sum() / g.w.sum(),
        "dcad_extrnl_depr": g.extrnl_x.sum() / g.w.sum(),
    }).reset_index()
    cols = [c for c in out.columns if c != "universal_id"]
    return df[["universal_id"]].merge(out, on="universal_id", how="left")[cols]


# --------------------------------------------------------------------------- owners

_UNIT = re.compile(r"\b(STE|SUITE|UNIT|APT|FL|FLOOR|RM|ROOM|BLDG|PMB)\b\s*\w+")


def sponsor_key(df: pd.DataFrame) -> pd.Series:
    """Owner mailing address + ZIP5, normalized. Single-asset LLCs under one sponsor
    usually share a mailing address even when their names differ."""
    raw = df.owner_mailing_address.replace({"NA": np.nan, "": np.nan})
    mail = (
        raw.str.upper()
        .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
        .str.replace(_UNIT, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    z = df.owner_zip.replace({"NA": np.nan, "": np.nan}).astype("string").str[:5]
    return (mail + " " + z.fillna("")).where(mail.notna() & (mail != ""))


def owner_keys(df: pd.DataFrame) -> pd.DataFrame:
    owner = df.owner_key.replace({"": np.nan, "NA": np.nan})
    sponsor = sponsor_key(df)
    return pd.DataFrame({
        "owner_group": owner,
        "sponsor_group": sponsor,
        "log_sponsor_portfolio": np.log(sponsor.map(sponsor.value_counts())),
    })


def _target_encode(keys: np.ndarray, y: np.ndarray, tr: np.ndarray, a: float = 5.0):
    """Smoothed group bottom-10 rate from training labels.

    Test rows get full-training-set statistics; training rows get 5-fold out-of-fold
    statistics (leave-one-out leaks a row's own label to tree models).
    """
    n_rows = len(keys)
    rate = np.full(n_rows, np.nan)
    count = np.zeros(n_rows)
    prior = y[tr].mean()

    def encode(fit_idx: np.ndarray, apply_idx: np.ndarray) -> None:
        s = (
            pd.DataFrame({"k": keys[fit_idx], "y": y[fit_idx]})
            .dropna(subset=["k"])
            .groupby("k").y.agg(["sum", "count"])
        )
        k = pd.Series(keys[apply_idx])
        su = k.map(s["sum"]).fillna(0).to_numpy()
        n = k.map(s["count"]).fillna(0).to_numpy()
        rate[apply_idx] = (su + a * prior) / (n + a)
        count[apply_idx] = n

    encode(tr, np.setdiff1d(np.arange(n_rows), tr))
    for fit_i, apply_i in KFold(5, shuffle=True, random_state=SEED).split(tr):
        encode(tr[fit_i], tr[apply_i])
    return rate, count


def owner_track_record(d: pd.DataFrame, tr: np.ndarray, target: str) -> pd.DataFrame:
    """Fold feature: how the owner's and sponsor's other inspected properties scored.

    Uses a `costar_owner` column (CoStar true owner) as a third grouping when present."""
    y = d[target].astype(int).to_numpy()
    keys = owner_keys(d)
    groups = ["owner_group", "sponsor_group"]
    if "costar_owner" in d:
        keys["costar_owner"] = d.costar_owner.astype(object).where(d.costar_owner.notna())
        groups.append("costar_owner")
    out = {}
    for name in groups:
        rate, count = _target_encode(keys[name].to_numpy(dtype=object), y, tr)
        out[f"{name}_b10_rate"] = rate
        out[f"{name}_n_seen"] = count
    return pd.DataFrame(out, index=d.index)


# --------------------------------------------------------------------------- space

def _radians(d: pd.DataFrame) -> np.ndarray:
    return np.radians(d[["latitude", "longitude"]].to_numpy(dtype=float))


def spatial_context(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free surroundings: multitenant density and distance to downtown."""
    xy = _radians(df)
    tree = BallTree(xy, metric="haversine")
    units = df.n_units.fillna(0).to_numpy()
    out = {}
    for km in (0.5, 1.5):
        idx = tree.query_radius(xy, r=km / EARTH_KM)
        out[f"n_props_within_{km}km"] = [len(i) - 1 for i in idx]
        out[f"log_units_within_{km}km"] = np.log1p([units[i].sum() - u for i, u in zip(idx, units)])
    cbd = np.radians(np.array([DALLAS_CBD]))
    out["km_to_downtown"] = BallTree(cbd, metric="haversine").query(xy)[0][:, 0] * EARTH_KM
    return pd.DataFrame(out, index=df.index)


def nearby_outcomes(d: pd.DataFrame, tr: np.ndarray, target: str, k: int = 10) -> pd.DataFrame:
    """Fold feature: bottom-10 rate of the k nearest inspected (training) properties,
    and how far away they are. A training row never counts itself."""
    y = d[target].astype(int).to_numpy()
    xy = _radians(d)
    tree = BallTree(xy[tr], metric="haversine")
    dist, idx = tree.query(xy, k=k + 1)
    # exclude the row itself wherever it appears (properties can share coordinates, so
    # the self-match is not always first), then keep the k nearest that remain
    is_self = tr[idx] == np.arange(len(d))[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")[:, :k]
    dist = np.take_along_axis(dist, order, axis=1) * EARTH_KM
    labels = y[tr][np.take_along_axis(idx, order, axis=1)]
    return pd.DataFrame({
        "knn_b10_rate": labels.mean(axis=1),
        "knn_nearest_km": dist[:, 0],
        "knn_mean_km": dist.mean(axis=1),
    }, index=d.index)


def combine(*builders):
    """Chain several fold builders into one."""
    def fold(d: pd.DataFrame, tr: np.ndarray, target: str) -> pd.DataFrame:
        return pd.concat([b(d, tr, target) for b in builders], axis=1)
    return fold


# --------------------------------------------------------------------------- supplemental sources
#
# Timing classes, relative to the inspection window (FY2020-21 to FY2024-25):
#   pre      measured before 2020-10-01; clean under any design
#   during   measured inside the window, concurrent with the outcomes
#   snapshot a 2026 snapshot taken after the window

WINDOW_START = pd.Timestamp("2020-10-01")
# 2025 DCAD parcel centroids (EPSG:2276, US feet) from
# dev_bronze.bronze_tx_dallas_cad.parcel_geom_certified, copied into _data/.
CENTROIDS = "dcad_parcel_centroids_2025.parquet"
WINDOW_END = pd.Timestamp("2025-10-01")


def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.upper()
        .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
        .str.replace(r"\b(LLC|LP|LTD|INC|CORP|CO|THE|L L C|L P)\b", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip()
        .replace("", pd.NA)
    )


def _per_100_units(count: pd.Series | np.ndarray, units: pd.Series, years: float) -> np.ndarray:
    return np.log1p(100 * np.asarray(count, dtype=float) / units.where(units > 0).to_numpy() / years)


def evictions(df: pd.DataFrame, data: Path = DATA, centroids: Path | None = None,
              radius_ft: float = 600.0) -> pd.DataFrame:
    """Eviction filings resolved to the property's DCAD parcel, per 100 units per year.

    Filings at large complexes often geocode to the street entrance and land on a
    neighbouring parcel. Following the resolution method's finding that only "adjacent and
    same owner" expansion beats doing nothing, filings on parcels within `radius_ft` whose
    owner of record matches the property's DCAD owner are added (`_adj` columns).
    """
    ev = pd.read_parquet(data / "evictions_parcel_year.parquet")
    pm = pd.read_csv(data / "parcel_match.csv", usecols=["address", "gis_parcel_id", "owner_name"], dtype=str)
    prop = df[["address", "n_units"]].merge(pm, on="address", how="left")
    units = prop.n_units

    windows = {"pre": (2017, 2020), "during": (2021, 2025)}
    by_parcel = {
        (w, strong): ev[ev.filing_year.between(*yrs)].groupby("parcel_id")["n_cases_strong" if strong else "n_cases"].sum()
        for w, yrs in windows.items() for strong in (False, True)
    }
    out = {}
    for (w, strong), s in by_parcel.items():
        years = windows[w][1] - windows[w][0] + 1
        cnt = prop.gis_parcel_id.map(s).fillna(0).where(prop.gis_parcel_id.notna())
        out[f"ev_{w}{'_strong' if strong else ''}"] = _per_100_units(cnt, units, years)

    # adjacent same-owner expansion
    cpath = centroids or data / CENTROIDS
    cen = pd.read_parquet(cpath).drop_duplicates("gis_parcel_id").set_index("gis_parcel_id")[["x", "y"]]
    ev_owner = ev.assign(owner=_norm_name(ev.owner_name)).dropna(subset=["owner"])
    ev_parcels = cen.reindex(ev_owner.parcel_id.unique()).dropna()
    tree = BallTree(ev_parcels.to_numpy())
    here = cen.reindex(prop.gis_parcel_id)
    ok = here.notna().all(axis=1).to_numpy()
    nbrs = np.empty(len(prop), dtype=object)
    nbrs[ok] = tree.query_radius(here[ok].to_numpy(), r=radius_ft)
    names = ev_parcels.index.to_numpy()
    prop_owner = _norm_name(prop.owner_name)
    for w, yrs in windows.items():
        sub = ev_owner[ev_owner.filing_year.between(*yrs)]
        cnt_by = sub.groupby(["parcel_id", "owner"]).n_cases.sum()
        adj = np.full(len(prop), np.nan)
        for i in np.flatnonzero(ok):
            own, me = prop_owner.iat[i], prop.gis_parcel_id.iat[i]
            # the property's own parcel is counted separately below
            adj[i] = 0.0 if pd.isna(own) else sum(
                cnt_by.get((p, own), 0.0) for p in names[nbrs[i]] if p != me
            )
        own_cnt = prop.gis_parcel_id.map(by_parcel[(w, False)]).fillna(0).where(prop.gis_parcel_id.notna()).to_numpy()
        years = yrs[1] - yrs[0] + 1
        out[f"ev_{w}_adj"] = _per_100_units(np.where(ok, own_cnt + np.nan_to_num(adj), own_cnt), units, years)
    return pd.DataFrame(out, index=df.index)


def eviction_parcel_sets(df: pd.DataFrame, data: Path = DATA, centroids: Path | None = None,
                         radius_ft: float = 600.0) -> list[list[str]]:
    """Per property: its own DCAD parcel plus parcels within `radius_ft` where any filing
    names the property's DCAD owner (the "adjacent and same owner" rule). Empty when the
    property has no parcel. Used for time-windowed counts in the temporal design."""
    ev = pd.read_parquet(data / "evictions_parcel_year.parquet")
    pm = pd.read_csv(data / "parcel_match.csv", usecols=["address", "gis_parcel_id", "owner_name"], dtype=str)
    prop = df[["address"]].merge(pm, on="address", how="left")
    cpath = centroids or data / CENTROIDS
    cen = pd.read_parquet(cpath).drop_duplicates("gis_parcel_id").set_index("gis_parcel_id")[["x", "y"]]
    owners_by_parcel = (
        ev.assign(owner=_norm_name(ev.owner_name)).dropna(subset=["owner"])
        .groupby("parcel_id").owner.agg(set)
    )
    ev_parcels = cen.reindex(owners_by_parcel.index).dropna()
    tree = BallTree(ev_parcels.to_numpy())
    names = ev_parcels.index.to_numpy()
    prop_owner = _norm_name(prop.owner_name)
    here = cen.reindex(prop.gis_parcel_id)
    sets = []
    for i in range(len(prop)):
        me = prop.gis_parcel_id.iat[i]
        if pd.isna(me):
            sets.append([])
            continue
        s = [me]
        own = prop_owner.iat[i]
        if pd.notna(own) and here.iloc[i].notna().all():
            for p in names[tree.query_radius(here.iloc[[i]].to_numpy(), r=radius_ft)[0]]:
                if p != me and own in owners_by_parcel.get(p, ()):
                    s.append(p)
        sets.append(s)
    return sets


def dcad_history(df: pd.DataFrame, data: Path = DATA) -> pd.DataFrame:
    """Value trajectory, sales, owner turnover and tax-protest agents, 2016-2021 (pre)."""
    acct = pd.read_parquet(data / "dcad_account.parquet")
    deed = pd.read_parquet(data / "dcad_deed.parquet")
    owner = pd.read_parquet(data / "dcad_owner.parquet")
    uid = df.universal_id

    val = acct.pivot_table(index="universal_id", columns="appraisal_year", values="total_market_value", aggfunc="sum")
    v = lambda y: val.get(y).where(val.get(y) > 0) if y in val else pd.Series(np.nan, index=val.index)
    pre = acct[acct.appraisal_year <= 2021]
    rep = pre.groupby("universal_id").has_taxpayer_rep.mean()
    pen = pre.assign(p=pre.rendition_penalty.fillna("").str.strip().ne("")).groupby("universal_id").p.max()

    deed = deed.assign(d=pd.to_datetime(deed.deed_transfer_date, format="%m/%d/%Y", errors="coerce"))
    deed = deed[deed.d > "1902-01-01"]
    pre_sales = deed[deed.d < WINDOW_START].groupby("universal_id").d
    during_sales = deed[(deed.d >= WINDOW_START) & (deed.d < WINDOW_END)].groupby("universal_id").d.nunique()
    n_sales_16_20 = deed[(deed.d >= "2016-01-01") & (deed.d < WINDOW_START)].groupby("universal_id").d.nunique()

    o1 = owner[(owner.owner_seq == owner.groupby(["universal_id", "appraisal_year"]).owner_seq.transform("min"))]
    o1 = o1.assign(n=_norm_name(o1.owner_name))
    owners_pre = o1[o1.appraisal_year <= 2021].groupby("universal_id").n.nunique()

    feats = pd.DataFrame({
        "value_growth_16_21": np.log(v(2021) / v(2016)),
        "value_growth_19_21": np.log(v(2021) / v(2019)),
        "share_years_taxpayer_rep": rep,
        "any_rendition_penalty": pen.astype(float),
        "years_since_sale_2020": (WINDOW_START - pre_sales.max()).dt.days / 365.25,
        "n_sales_16_20": n_sales_16_20,
        "n_owner_names_16_21": owners_pre,
        "sold_during_window": (during_sales > 0).astype(float),
    })
    out = feats.reindex(uid.to_numpy())
    out.index = df.index
    known = uid.isin(acct.universal_id).to_numpy()
    for c in ["n_sales_16_20", "sold_during_window"]:
        out[c] = out[c].fillna(0).where(known)
    return out


def tax_delinquency(df: pd.DataFrame, data: Path = DATA) -> pd.DataFrame:
    """DCTO roll snapshot (Aug-Sep 2026). Bills cured before the snapshot are invisible, so
    `delinquent_since_pre2021` (still unpaid from 2020 or earlier) is the only part that is
    known to predate the window."""
    t = pd.read_parquet(data / "tax_summary.parquet").set_index("universal_id")
    t = t.reindex(df.universal_id.to_numpy())
    t.index = df.index
    known = df.universal_id.notna()
    deliq = t.is_delinquent.astype("boolean")
    return pd.DataFrame({
        "delinquent_since_pre2021": (deliq & (t.oldest_delinquent_year <= 2020)).astype(float).where(known),
        "delinquent_2026": deliq.astype(float).where(known),
        "n_delinquent_years_2026": t.num_delinquent_years.fillna(0).where(known),
        "tax_suit_2026": t.any_suit.astype("boolean").astype(float).where(known),
    }, index=df.index)


def costar(df: pd.DataFrame, data: Path = DATA, radius_m: float = 75.0) -> tuple[pd.DataFrame, pd.Series]:
    """CoStar multifamily (2026-03 snapshot): class, star rating, affordability, rent, size.
    Matched on DCAD account, else nearest CoStar point within `radius_m`. Also returns the
    CoStar true-owner name as a sponsor key for the owner track record."""
    c = pd.read_parquet(data / "costar_mf.parquet")
    key = c.parcel_number_min.str.replace(r"[^0-9A-Za-z]", "", regex=True).str.upper()
    c = c.assign(account=key).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    acct = df.universal_id.str[6:]
    by_acct = c.drop_duplicates("account").set_index("account")
    hit = acct.map(lambda a: a in by_acct.index if isinstance(a, str) else False)
    row = pd.Series(np.nan, index=df.index)
    row[hit] = c.reset_index().drop_duplicates("account").set_index("account")["index"].reindex(acct[hit]).to_numpy()

    tree = BallTree(np.radians(c[["latitude", "longitude"]].to_numpy()), metric="haversine")
    miss = ~hit & df.latitude.notna()
    dist, idx = tree.query(np.radians(df.loc[miss, ["latitude", "longitude"]].to_numpy(dtype=float)), k=1)
    near = dist[:, 0] * EARTH_KM * 1000 <= radius_m
    row[miss[miss].index[near]] = idx[near, 0]

    m = c.reindex(row.to_numpy())
    m.index = df.index
    cls = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 1}
    out = pd.DataFrame({
        "costar_class": m.building_class.map(cls),
        "costar_stars": pd.to_numeric(m.star_rating, errors="coerce"),
        "costar_affordable": m.affordable_type.notna().astype(float).where(row.notna()),
        "costar_log_rent_per_unit": np.log(m.avg_asking_unit.where(m.avg_asking_unit > 0)),
        "costar_unit_sf": m.avg_unit_sf,
        "costar_vacancy_2026": m.vacancy_percent,
        "costar_years_since_renovated": 2021 - pd.to_numeric(m.year_renovated, errors="coerce").where(lambda y: y <= 2021),
    }, index=df.index)
    true_owner = _norm_name(m.true_owner_name.fillna(m.owner_name))
    return out, true_owner


def nearby_points(df: pd.DataFrame, pts: pd.DataFrame, radius_m: float, units_norm: bool = True,
                  years: float = 1.0) -> np.ndarray:
    """Count of points within `radius_m` of each property, optionally per 100 units per year."""
    pts = pts.dropna(subset=["latitude", "longitude"])
    if pts.empty:
        counts = np.zeros(len(df))
    else:
        tree = BallTree(np.radians(pts[["latitude", "longitude"]].to_numpy(dtype=float)), metric="haversine")
        counts = tree.query_radius(np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float)),
                                   r=radius_m / 1000 / EARTH_KM, count_only=True)
    return _per_100_units(counts, df.n_units, years) if units_norm else counts


def apartment_crime(df: pd.DataFrame, data: Path = DATA, radius_m: float = 150.0) -> pd.DataFrame:
    """DPD NIBRS incidents at apartment premises (2023-01 onward; concurrent, partial window)."""
    n = pd.read_parquet(data / "nibrs_apartment.parquet")
    n["date_event"] = pd.to_datetime(n.date_event)
    n = n[(n.date_event >= "2023-01-01") & (n.date_event < WINDOW_END)]
    years = (WINDOW_END - pd.Timestamp("2023-01-01")).days / 365.25
    violent = n.category.astype("string").str.contains("ASSAULT|HOMICIDE|ROBBERY|KIDNAP|SEX", case=False, na=False)
    return pd.DataFrame({
        "crime_apartment_2023_25": nearby_points(df, n, radius_m, years=years),
        "crime_violent_2023_25": nearby_points(df, n[violent], radius_m, years=years),
    }, index=df.index)


def complaints_311(df: pd.DataFrame, data: Path = DATA, radius_m: float = 100.0) -> pd.DataFrame:
    """Resident-initiated 311 Code Compliance requests near the property, per 100 units per
    year. Excludes city-initiated requests (method Proactive or Internal) and any request
    whose number matches a graded inspection, so the inspections never count themselves.

    The specific hazard types (A/C or heat inoperable, life hazard, substandard structure,
    rodent, open and vacant) exist only through 2019; from 2020 the city files nearly all of
    them as the generic "Code Concern - CCS". So the hazard subset is pre-window only, and
    Emergency priority stands in as the severity cut for both windows."""
    r = pd.read_parquet(data / "311_code_compliance.parquet")
    insp = pd.read_stata(data / "multitenant_inspections.dta", convert_categoricals=False)
    r = r[~r.service_request_number.isin(set(insp.sr_number.astype(str)))]
    r = r[~r.method_received_description.isin(["Proactive", "Internal"])]
    hazard = r.service_request_type.astype("string").str.contains(
        "Heat Inoperable|Life Hazard|Substandard|Rodent|Open and Vacant", case=False, na=False
    )
    emergency = r.priority.astype("string").eq("Emergency").fillna(False)
    out = {}
    for w, (a, b) in {"pre": ("2017-01-01", WINDOW_START), "during": (WINDOW_START, WINDOW_END)}.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        in_w = (r.created_date >= a) & (r.created_date < b)
        years = (b - a).days / 365.25
        out[f"c311_{w}"] = nearby_points(df, r[in_w], radius_m, years=years)
        out[f"c311_emergency_{w}"] = nearby_points(df, r[in_w & emergency], radius_m, years=years)
        if w == "pre":
            out["c311_hazard_pre"] = nearby_points(df, r[in_w & hazard], radius_m, years=years)
    return pd.DataFrame(out, index=df.index)
