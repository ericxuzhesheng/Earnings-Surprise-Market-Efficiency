import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import ProjectConfig
from src.data_collection import DataCollector
from src.event_validation import calendar_returns, clustered_estimate, has_contamination, match_consensus, window_car
from src.expectation_alignment import _candidate_universe_for_tier
from src.tushare_normalization import normalize_fina_indicator, normalize_forecast, normalize_report_rc


def test_announcement_cutoff_excludes_same_day_and_weekend_news():
    events = pd.DataFrame([dict(event_id=1, ts_code="A", ann_date=pd.Timestamp("2024-01-05"),
                               event_trade_date=pd.Timestamp("2024-01-08"), period_end=pd.Timestamp("2023-12-31"),
                               forecast_profit_mid=130)])
    reports = pd.DataFrame(dict(ts_code=["A"] * 5,
                               period_end=pd.to_datetime(["2023-12-31"] * 4 + ["2024-12-31"]),
                               report_date=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-04"]),
                               org_name=["broker1", "broker1", "broker1", "broker2", "broker3"],
                               np=[90, 100, 130, 130, 999]))
    result = match_consensus(events, reports).iloc[0]
    assert result.expected_np == 100  # Latest per broker; future fiscal year excluded.
    assert result.matched_brokers == 1
    assert result.surprise_pct == pytest.approx(.3)
    assert match_consensus(events, reports, before_announcement=False).iloc[0].expected_np == 130
    candidates = _candidate_universe_for_tier(events.iloc[0], reports, ProjectConfig(), "strict_same_quarter")
    assert len(candidates) == 2
    assert candidates.report_date.max() < events.ann_date.iloc[0]


def test_calendar_windows_do_not_compress_missing_sessions_or_partial_tails():
    calendar = pd.bdate_range("2024-01-01", periods=7)
    market = pd.DataFrame({"trade_date": calendar, "mkt_ret": 0.0})
    prices = pd.DataFrame({"trade_date": calendar.delete(3), "close": [100, 101, 102, 104, 105, 106]})
    result = calendar_returns(prices, market)
    assert len(result) == 7
    assert pd.isna(result.ret.iloc[3]) and pd.isna(result.ret.iloc[4])
    assert pd.isna(window_car(result, calendar[1], 1, 3))
    assert pd.isna(window_car(result, calendar[-1], 1, 10))
    assert window_car(result, calendar[1], 0, 1) == pytest.approx(.01 + 102 / 101 - 1)


def test_split_adjustment_and_missing_factors_are_not_zero_returns():
    calendar = pd.bdate_range("2024-01-01", periods=4)
    market = pd.DataFrame({"trade_date": calendar, "mkt_ret": 0.0})
    prices = pd.DataFrame({"trade_date": calendar, "close": [100, 102, 51, 52]})
    factors = pd.DataFrame({"trade_date": calendar, "adj_factor": [1, 1, 2, 2]})
    result = calendar_returns(prices, market, factors)
    assert result.ret.iloc[2] == 0
    assert calendar_returns(prices, market).ret.iloc[2] == -.5
    factors.loc[2, "adj_factor"] = np.nan
    assert pd.isna(calendar_returns(prices, market, factors).ret.iloc[2])


def test_earnings_overlap_uses_exchange_sessions():
    calendar = pd.bdate_range("2024-01-01", periods=50)
    assert has_contamination(calendar[12], [calendar[12], calendar[20]], calendar, 10)
    assert not has_contamination(calendar[12], [calendar[12], calendar[30]], calendar, 10)


def test_profit_is_not_target_price_or_comparable_deducted_profit():
    reports = pd.DataFrame([dict(ts_code="A", report_date="20240101", quarter="2023Q4",
                                np=100, eps=1, pe=10, tp=20000, report_title="forecast")])
    assert pd.isna(normalize_report_rc(reports).target_price_mid.iloc[0])
    fina = pd.DataFrame([dict(ts_code="A", ann_date="20240101", end_date="20231231", profit_dedt=1e8, eps=1)])
    assert pd.isna(normalize_fina_indicator(fina).actual_np.iloc[0])


def test_report_pagination_exhausts_and_rejects_repeated_page(monkeypatch):
    collector = DataCollector.__new__(DataCollector)
    collector.config = ProjectConfig(request_pause_sec=0)
    collector.logger = logging.getLogger("test")
    monkeypatch.setattr("src.data_collection.time.sleep", lambda _: None)
    offsets = []

    def query(**kwargs):
        offsets.append(kwargs["offset"])
        return pd.DataFrame({"row": [1, 2]}) if kwargs["offset"] == 0 else pd.DataFrame()

    collector.ts = SimpleNamespace(report_rc=query)
    assert len(collector._get_complete_report_month("20240101", "20240131")) == 2
    assert offsets == [0, 2]  # Even a short first page must be exhausted.
    collector.ts = SimpleNamespace(report_rc=lambda **kw: pd.DataFrame({"row": [1]}))
    with pytest.raises(ValueError, match="repeated"):
        collector._get_complete_report_month("20240101", "20240131")


def test_cluster_inference_refuses_too_few_clusters():
    data = pd.DataFrame({"ts_code": ["A"] * 20, "event_trade_date": pd.date_range("2020-01-01", periods=20),
                         "surprise_pct": [-1, 1] * 10, "car": [0, .02] * 10})
    result = clustered_estimate(data, "car", difference=True)
    assert result["estimate"] == pytest.approx(.02)
    assert not result["inference_valid"] and np.isnan(result["ci_low"])


def test_first_cached_row_can_already_be_a_revision():
    forecast = pd.DataFrame([dict(ts_code="A", ann_date="20240110", first_ann_date="20240105",
        end_date="20231231", update_flag=1, p_change_min=5, p_change_max=10,
        net_profit_min=100, net_profit_max=110)])
    assert normalize_forecast(forecast).is_revision.iloc[0]
