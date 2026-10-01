"""Pull Dallas 311 Code Compliance service requests (Dallas Open Data, dataset 4n3h-rfqm).

Writes `_data/311_code_compliance.parquet`: one row per service request, Code Compliance
department only, created 2017-01-01 through 2025-09-30, with type, method received,
priority, created date, address and coordinates. The parquet is gitignored.

Paced at one request every 3 seconds in 50,000-row CSV pages (about 25 pages).

    uv run python _code/pull_311.py
"""

from __future__ import annotations

import io
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://www.dallasopendata.com/resource/4n3h-rfqm.csv"
OUT = Path(__file__).resolve().parent.parent / "_data" / "311_code_compliance.parquet"
PAGE = 50_000
PAUSE_S = 3.0
COLUMNS = [
    "service_request_number", "service_request_type", "method_received_description",
    "priority", "status", "created_date", "address", "lat_long_location",
]
WHERE = (
    "department = 'Code Compliance' "
    "AND created_date >= '2017-01-01T00:00:00' AND created_date < '2025-10-01T00:00:00'"
)


def page(offset: int) -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "$select": ", ".join(COLUMNS),
        "$where": WHERE,
        "$order": "service_request_number",
        "$limit": PAGE,
        "$offset": offset,
    })
    with urllib.request.urlopen(f"{URL}?{q}", timeout=300) as r:
        return pd.read_csv(io.BytesIO(r.read()), dtype=str)


def parse_point(s: pd.Series) -> pd.DataFrame:
    # lat_long_location arrives as text such as "(32.78, -96.80)" or a WKT "POINT (lon lat)"
    def one(v: str | float):
        if not isinstance(v, str):
            return (None, None)
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", v)]
        if len(nums) < 2:
            return (None, None)
        a, b = nums[:2]
        return (a, b) if abs(a) < 40 else (b, a)

    pts = s.map(one)
    return pd.DataFrame({"latitude": [p[0] for p in pts], "longitude": [p[1] for p in pts]}, index=s.index)


def main() -> None:
    frames, offset = [], 0
    while True:
        df = page(offset)
        frames.append(df)
        print(f"offset {offset:>9,}: {len(df):,} rows", flush=True)
        if len(df) < PAGE:
            break
        offset += PAGE
        time.sleep(PAUSE_S)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("service_request_number")
    out = pd.concat([out, parse_point(out.lat_long_location)], axis=1).drop(columns="lat_long_location")
    out["created_date"] = pd.to_datetime(out.created_date)
    OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out):,} requests to {OUT}; {out.latitude.notna().mean():.1%} with coordinates")


if __name__ == "__main__":
    main()
