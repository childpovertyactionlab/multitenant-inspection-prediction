"""Who inspected each graded inspection, reconciled across the two sources.

1. Salesforce "Assessed By: Name" on the observation rows of
   `multitenant_observations.dta`. A system user field, so spellings are consistent. The
   inspection-level analytic file kept only the first row's name per inspection, which
   drops names that appear on later rows (34% coverage there, 62% here).
2. Code Compliance's July 2026 backfill (`code_inspector_backfill_2026.xlsx`). Typed by
   hand: case, initials, suffixes and name order vary.

Backfill spellings are mapped to Salesforce spellings from the data, never by hand: a
backfill name maps to a Salesforce name when at least `min_share` of the inspections both
sources name agree, at least `min_overlap` overlap, and the two names share a token.
Where both sources name an inspector, Salesforce wins and disagreements are flagged.

Outputs use stable codes (I01, I02, ... by inspection count) rather than names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from holdout import DATA

OBS = "multitenant_observations.dta"
OBS_CACHE = "observation_inspector_names.parquet"
BACKFILL = "code_inspector_backfill_2026.xlsx"


def norm(s: pd.Series) -> pd.Series:
    """Upper case, letters only, single spaces; blank becomes missing."""
    out = (
        s.astype("string").str.upper()
        .str.replace(r"[^A-Z ]", " ", regex=True)
        .str.split().str.join(" ")
    )
    return out.where(~out.isin(["", "NA"]))


def salesforce_names(data: Path = DATA) -> pd.DataFrame:
    """Per inspection: most frequent Salesforce name across its rows, and how many differ."""
    cache = data / OBS_CACHE
    if cache.exists():
        rows = pd.read_parquet(cache)
    else:
        parts = []
        with pd.read_stata(data / OBS, iterator=True, chunksize=20_000,
                           columns=["sr_number", "assessor_name"]) as r:
            parts.extend(r)
        rows = pd.concat(parts, ignore_index=True)
        rows.to_parquet(cache, index=False)
    rows = rows.assign(sr_number=rows.sr_number.astype(str).str.strip(), name=norm(rows.assessor_name))
    named = rows.dropna(subset=["name"])
    g = named.groupby("sr_number").name
    return pd.DataFrame({"sf_name": g.agg(lambda s: s.mode().iat[0]), "sf_n_names": g.nunique()})


def backfill_names(data: Path = DATA) -> pd.Series:
    x = pd.read_excel(data / BACKFILL, usecols=["sr_number", "Inspector"])
    x = x.assign(sr_number=x.sr_number.astype(str).str.strip(), name=norm(x.Inspector)).dropna(subset=["name"])
    return x.drop_duplicates("sr_number").set_index("sr_number").name.rename("bf_name")


def alias_map(sf: pd.Series, bf: pd.Series, min_share: float = 0.9, min_overlap: int = 2) -> dict[str, str]:
    """Backfill spelling -> Salesforce spelling, learned from inspections both sources name."""
    both = pd.concat([sf.rename("sf"), bf.rename("bf")], axis=1, join="inner")
    tab = both.groupby(["bf", "sf"]).size().rename("n").reset_index()
    tab["share"] = tab.n / tab.groupby("bf").n.transform("sum")
    sf_names = set(sf.unique())
    out = {}
    for row in tab.itertuples():
        if row.bf == row.sf or row.bf in sf_names:
            continue
        shared = {w for w in row.bf.split() if len(w) >= 3} & set(row.sf.split())
        if row.share >= min_share and row.n >= min_overlap and shared:
            out[row.bf] = row.sf
    return out


def inspector_table(data: Path = DATA) -> pd.DataFrame:
    """One row per inspection (sr_number): canonical inspector, source, code, conflict flag."""
    sf = salesforce_names(data)
    bf = backfill_names(data)
    aliases = alias_map(sf.sf_name, bf)
    bf_canon = bf.replace(aliases)
    t = sf.join(bf_canon.rename("bf_canon"), how="outer").join(bf, how="left")
    t["inspector_name"] = t.sf_name.fillna(t.bf_canon)
    t["source"] = np.select([t.sf_name.notna(), t.bf_canon.notna()], ["salesforce", "backfill"], "none")
    t["conflict"] = t.sf_name.notna() & t.bf_canon.notna() & (t.sf_name != t.bf_canon)
    counts = t.inspector_name.value_counts()
    code = {n: f"I{k:02d}" for k, n in enumerate(counts.index, start=1)}
    t["inspector_id"] = t.inspector_name.map(code)
    t.attrs["aliases"] = aliases
    return t.rename_axis("sr_number").reset_index()
