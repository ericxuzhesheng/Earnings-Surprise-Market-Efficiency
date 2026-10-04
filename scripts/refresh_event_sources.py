"""Repair only the frozen 300-stock sample, never extend its date range."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/event_validation"


def run(endpoint):
    token = os.environ["TUSHARE_TOKEN"]
    universe = pd.read_csv(ROOT / "data_raw/cache/stock_basic/listed.csv").sort_values("ts_code").head(300)
    if endpoint == "report_rc":
        codes = universe.ts_code.tolist()
    else:
        matched = pd.read_csv(OUT / "matching_audit.csv")
        codes = sorted(matched.loc[matched.surprise_pct.notna(), "ts_code"].unique())
    target = OUT / f"source_{endpoint}"
    target.mkdir(parents=True, exist_ok=True)

    def fetch(code):
        path = target / f"{code}.csv"
        if path.exists():
            return
        client = ts.pro_api(token)
        pages, offset, previous = [], 0, None
        while True:
            if endpoint in {"daily_basic", "fina_indicator"}:
                time.sleep(1.6)  # Four workers remain below the 200/min endpoint cap.
            params = dict(ts_code=code, start_date="20200101", end_date="20260421", limit=3000, offset=offset)
            for attempt in range(3):
                try:
                    data = client.query(endpoint, **params)
                    break
                except Exception as exc:
                    if any(word in str(exc) for word in ["频次", "权限", "积分"]):
                        raise  # Respect provider rate/access limits; do not hammer retries.
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            if data.empty:
                break
            if previous is not None and previous.equals(data):
                raise ValueError(f"{endpoint} repeated page for {code}")
            pages.append(data)
            previous = data
            offset += len(data)
        data = pd.concat(pages, ignore_index=True).drop_duplicates() if pages else pd.DataFrame({"ts_code": []})
        data.to_csv(path, index=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, codes))
    (target / "receipt.json").write_text(json.dumps({"endpoint": endpoint, "codes": codes,
        "start_date": "20200101", "end_date": "20260421", "pagination": "until empty page",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_in_time_vintage": False}, indent=2), encoding="utf-8")
    print(f"{endpoint}: {len(codes)} complete stock snapshots", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", choices=["report_rc", "daily", "daily_basic", "fina_indicator"])
    run(parser.parse_args().endpoint)
