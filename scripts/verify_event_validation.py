"""Independent arithmetic checks against saved provider inputs, without model helpers."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/event_validation"


def main():
    events = pd.read_csv(OUT / "events.csv", parse_dates=["event_trade_date", "ann_date", "latest_report_date"])
    market = pd.read_csv(ROOT / "data_raw/cache/market/market_399300_SZ_20200101_20260421.csv")
    market["trade_date"] = pd.to_datetime(market.trade_date.astype(str), format="mixed")
    market = market.sort_values("trade_date").set_index("trade_date")
    market_return = market.close / market.close.shift(1) - 1
    checked = 0
    for code, group in events.groupby("ts_code"):
        raw = pd.read_csv(OUT / "source_daily" / f"{code}.csv")
        factors = pd.read_csv(OUT / "source_factors" / f"{code}.csv")
        raw["trade_date"] = pd.to_datetime(raw.trade_date.astype(str), format="%Y%m%d")
        factors["trade_date"] = pd.to_datetime(factors.trade_date.astype(str), format="%Y%m%d")
        close = raw.set_index("trade_date").close.reindex(market.index)
        factor = factors.set_index("trade_date").adj_factor.reindex(market.index)
        adjusted = close * factor
        ar = adjusted / adjusted.shift(1) - 1 - market_return
        for event in group.itertuples():
            pos = market.index.get_loc(event.event_trade_date)
            values = ar.iloc[pos + 1:pos + 11]
            expected = values.sum() if len(values) == 10 and values.notna().all() else np.nan
            np.testing.assert_allclose(event.CAR_1_10, expected, rtol=0, atol=1e-12, equal_nan=True)
            assert event.latest_report_date < event.ann_date < event.event_trade_date
            checked += 1
    clean = events.loc[events.eligible & ~events.contaminated_10].dropna(subset=["CAR_1_10", "surprise_pct"])
    positive = clean.loc[clean.surprise_pct > 0, "CAR_1_10"]
    negative = clean.loc[clean.surprise_pct < 0, "CAR_1_10"]
    table = pd.read_csv(OUT / "inference.csv")
    primary = table.loc[table.primary_test].iloc[0]
    np.testing.assert_allclose(primary.estimate, positive.mean() - negative.mean(), rtol=0, atol=1e-12)
    assert primary.n == len(positive) + len(negative)
    # Verify source completeness receipts rather than merely checking file presence.
    for endpoint in ["daily", "daily_basic", "fina_indicator"]:
        import json
        receipt = json.loads((OUT / f"source_{endpoint}/receipt.json").read_text())
        assert set(events.ts_code).issubset(receipt["codes"])
    print(f"Verified {checked} event CAR[1,10] calculations and chronological cutoffs; primary difference N={primary.n} reconciles.")


if __name__ == "__main__":
    main()
