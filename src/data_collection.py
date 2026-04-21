from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.io_utils import save_csv


@dataclass
class DataBundle:
    stocks: pd.DataFrame
    prices: pd.DataFrame
    market: pd.DataFrame
    daily_basic: pd.DataFrame
    forecast: pd.DataFrame
    express: pd.DataFrame
    fina_indicator: pd.DataFrame
    report_rc: pd.DataFrame
    capabilities: pd.DataFrame


class DataCollector:
    """Collect A-share data with Tushare as the preferred source."""

    def __init__(self, config: ProjectConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.ts = None
        self.ak = None
        self._load_clients()
        self.cache_root = self.config.data_raw_dir / "cache"
        for folder in [
            "forecast",
            "forecast_vip",
            "express",
            "express_vip",
            "fina_indicator",
            "report_rc",
            "prices",
            "daily_basic",
            "market",
            "stock_basic",
        ]:
            (self.cache_root / folder).mkdir(parents=True, exist_ok=True)

    def _load_clients(self) -> None:
        if self.config.tushare_token:
            try:
                import tushare as ts  # type: ignore

                self.ts = ts.pro_api(self.config.tushare_token)
                self.logger.info("Tushare client initialized.")
            except Exception as exc:
                self.logger.warning("Tushare unavailable: %s", exc)
        else:
            self.logger.warning("TUSHARE_TOKEN is not set; Tushare client not initialized.")

        try:
            import akshare as ak  # type: ignore

            self.ak = ak
            self.logger.info("Akshare client initialized.")
        except Exception as exc:
            self.logger.warning("Akshare unavailable: %s", exc)

    def collect_all(self) -> DataBundle:
        stocks = self.get_stock_universe()
        prices = self.get_stock_prices(stocks)
        market = self.get_market_index()
        daily_basic = self.get_daily_basic(stocks)
        forecast = self.get_forecast_data(stocks)
        express = self.get_express_data(stocks)
        fina_indicator = self.get_fina_indicator_data(stocks)
        report_rc = self.get_report_rc_data()
        capabilities = self.get_endpoint_capabilities()

        save_csv(stocks, self.config.data_raw_dir / "stock_universe.csv")
        save_csv(prices, self.config.data_raw_dir / "stock_prices_raw.csv")
        save_csv(market, self.config.data_raw_dir / "market_index_raw.csv")
        save_csv(daily_basic, self.config.data_raw_dir / "daily_basic_raw.csv")
        save_csv(forecast, self.config.data_raw_dir / "guidance_forecast_raw.csv")
        save_csv(express, self.config.data_raw_dir / "express_raw.csv")
        save_csv(fina_indicator, self.config.data_raw_dir / "fina_indicator_raw.csv")
        save_csv(report_rc, self.config.data_raw_dir / "report_rc_raw.csv")
        save_csv(capabilities, self.config.outputs_audit_dir / "tushare_endpoint_capabilities.csv")

        save_csv(stocks, self.config.data_raw_tushare_dir / "stock_basic" / "stock_universe.csv")
        save_csv(prices, self.config.data_raw_tushare_dir / "prices" / "stock_prices_raw.csv")
        save_csv(market, self.config.data_raw_tushare_dir / "market" / "market_index_raw.csv")
        save_csv(daily_basic, self.config.data_raw_tushare_dir / "daily_basic" / "daily_basic_raw.csv")
        save_csv(forecast, self.config.data_raw_tushare_dir / "forecast" / "forecast_raw.csv")
        save_csv(express, self.config.data_raw_tushare_dir / "express" / "express_raw.csv")
        save_csv(fina_indicator, self.config.data_raw_tushare_dir / "fina_indicator" / "fina_indicator_raw.csv")
        save_csv(report_rc, self.config.data_raw_tushare_dir / "report_rc" / "report_rc_raw.csv")

        return DataBundle(
            stocks=stocks,
            prices=prices,
            market=market,
            daily_basic=daily_basic,
            forecast=forecast,
            express=express,
            fina_indicator=fina_indicator,
            report_rc=report_rc,
            capabilities=capabilities,
        )

    def _cache_path(self, folder: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(".", "_")
        return self.cache_root / folder / f"{safe_key}.csv"

    def _load_cache(self, folder: str, key: str) -> pd.DataFrame | None:
        if not self.config.use_cache or self.config.force_refresh:
            return None
        p = self._cache_path(folder, key)
        if not p.exists():
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame, folder: str, key: str) -> None:
        if not self.config.use_cache:
            return
        p = self._cache_path(folder, key)
        try:
            df.to_csv(p, index=False, encoding="utf-8-sig")
        except Exception as exc:
            self.logger.warning("Cache save failed %s/%s: %s", folder, key, exc)

    def _retry(self, fn, context: str) -> pd.DataFrame | None:
        for i in range(self.config.max_retries):
            try:
                return fn()
            except Exception as exc:
                if i == self.config.max_retries - 1:
                    self.logger.warning("%s failed after retries: %s", context, exc)
                    return None
                time.sleep(self.config.retry_wait_seconds * (i + 1))
        return None

    @staticmethod
    def _parse_trade_date(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        need_fallback = out.isna()
        if need_fallback.any():
            out.loc[need_fallback] = pd.to_datetime(s.loc[need_fallback], errors="coerce")
        return out

    def get_endpoint_capabilities(self) -> pd.DataFrame:
        checks: list[tuple[str, callable]] = []
        if self.ts is not None:
            checks = [
                ("report_rc", lambda: self.ts.report_rc(start_date="20240101", end_date="20240131")),
                ("forecast", lambda: self.ts.forecast(ts_code="000001.SZ", start_date="20240101", end_date="20241231")),
                ("forecast_vip", lambda: self.ts.forecast_vip(period="20240331")),
                ("express", lambda: self.ts.express(ts_code="000001.SZ", start_date="20240101", end_date="20241231")),
                ("express_vip", lambda: self.ts.express_vip(period="20240331")),
                ("fina_indicator", lambda: self.ts.fina_indicator(ts_code="000001.SZ", start_date="20240101", end_date="20241231")),
                ("daily_basic", lambda: self.ts.daily_basic(ts_code="000001.SZ", start_date="20240101", end_date="20240131")),
            ]
        rows: list[dict[str, object]] = []
        for name, fn in checks:
            try:
                df = fn()
                rows.append({
                    "endpoint": name,
                    "available": True,
                    "row_count": int(len(df)),
                    "columns": ",".join(df.columns),
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "endpoint": name,
                    "available": False,
                    "row_count": 0,
                    "columns": "",
                    "error": str(exc),
                })
        return pd.DataFrame(rows)

    def get_stock_universe(self) -> pd.DataFrame:
        if self.ts is not None:
            cached = self._load_cache("stock_basic", "listed")
            if cached is not None and not cached.empty:
                df = cached.copy()
            else:
                try:
                    df = self.ts.stock_basic(
                        exchange="",
                        list_status="L",
                        fields="ts_code,symbol,name,area,industry,list_date",
                    )
                    if not df.empty:
                        self._save_cache(df, "stock_basic", "listed")
                except Exception as exc:
                    self.logger.warning("Tushare stock_basic failed: %s", exc)
                    df = pd.DataFrame()
            if not df.empty:
                df = df.sort_values("ts_code").reset_index(drop=True)
                if self.config.sample_stock_count:
                    df = df.head(self.config.sample_stock_count).copy()
                self.logger.info("Stock universe from Tushare: %s rows", len(df))
                return df

        if self.ak is not None:
            try:
                ak_df = self.ak.stock_info_a_code_name()
                ak_df = ak_df.rename(columns={"code": "symbol", "name": "name"})
                ak_df["ts_code"] = np.where(
                    ak_df["symbol"].str.startswith(("0", "3")),
                    ak_df["symbol"] + ".SZ",
                    ak_df["symbol"] + ".SH",
                )
                ak_df["industry"] = np.nan
                ak_df["area"] = np.nan
                ak_df["list_date"] = np.nan
                ak_df = ak_df[["ts_code", "symbol", "name", "area", "industry", "list_date"]]
                ak_df = ak_df.sort_values("ts_code").reset_index(drop=True)
                if self.config.sample_stock_count:
                    ak_df = ak_df.head(self.config.sample_stock_count).copy()
                self.logger.info("Stock universe from Akshare: %s rows", len(ak_df))
                return ak_df
            except Exception as exc:
                self.logger.error("Akshare stock universe failed: %s", exc)

        self.logger.error("No data source available for stock universe.")
        return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry", "list_date"])

    def get_forecast_data(self, stocks: pd.DataFrame) -> pd.DataFrame:
        if stocks.empty or self.ts is None:
            return pd.DataFrame()
        use_vip = self.config.use_forecast_vip
        folder = "forecast_vip" if use_vip else "forecast"
        rows: list[pd.DataFrame] = []
        codes = list(stocks["ts_code"].dropna().unique())
        if use_vip:
            periods = pd.period_range(
                start=pd.to_datetime(self.config.start_date, format="%Y%m%d"),
                end=pd.to_datetime(self.config.end_date, format="%Y%m%d"),
                freq="Q",
            )
            for period in periods:
                period_key = period.end_time.strftime("%Y%m%d")
                cached = self._load_cache(folder, period_key)
                if cached is not None:
                    if not cached.empty:
                        rows.append(cached)
                    continue
                fn = lambda pk=period_key: self.ts.forecast_vip(period=pk)
                df = self._retry(fn, f"forecast_vip {period_key}")
                if df is not None:
                    self._save_cache(df, folder, period_key)
                    if not df.empty:
                        rows.append(df)
                time.sleep(max(self.config.request_pause_sec, 0.12))
        else:
            iterator = codes if self.config.run_mode == "test" else codes
            for code in iterator:
                cached = self._load_cache(folder, code)
                if cached is not None:
                    if not cached.empty:
                        rows.append(cached)
                    continue
                fn = lambda c=code: self.ts.forecast(
                    ts_code=c,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                )
                df = self._retry(fn, f"forecast {code}")
                if df is not None:
                    self._save_cache(df, folder, code)
                    if not df.empty:
                        rows.append(df)
                time.sleep(max(self.config.request_pause_sec, 0.15))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        out = out[out["ts_code"].isin(codes)].copy()
        for col in ["ann_date", "end_date", "first_ann_date"]:
            if col in out.columns:
                out[col] = self._parse_trade_date(out[col])
        out["source_endpoint"] = folder
        out = out.sort_values([c for c in ["ts_code", "end_date", "ann_date"] if c in out.columns]).reset_index(drop=True)
        self.logger.info("Forecast rows collected: %s", len(out))
        return out

    def get_express_data(self, stocks: pd.DataFrame) -> pd.DataFrame:
        if stocks.empty or self.ts is None:
            return pd.DataFrame()
        use_vip = self.config.use_express_vip
        folder = "express_vip" if use_vip else "express"
        rows: list[pd.DataFrame] = []
        codes = list(stocks["ts_code"].dropna().unique())
        if use_vip:
            periods = pd.period_range(
                start=pd.to_datetime(self.config.start_date, format="%Y%m%d"),
                end=pd.to_datetime(self.config.end_date, format="%Y%m%d"),
                freq="Q",
            )
            for period in periods:
                period_key = period.end_time.strftime("%Y%m%d")
                cached = self._load_cache(folder, period_key)
                if cached is not None:
                    if not cached.empty:
                        rows.append(cached)
                    continue
                fn = lambda pk=period_key: self.ts.express_vip(period=pk)
                df = self._retry(fn, f"express_vip {period_key}")
                if df is not None:
                    self._save_cache(df, folder, period_key)
                    if not df.empty:
                        rows.append(df)
                time.sleep(max(self.config.request_pause_sec, 0.12))
        else:
            for code in codes:
                cached = self._load_cache(folder, code)
                if cached is not None:
                    if not cached.empty:
                        rows.append(cached)
                    continue
                fn = lambda c=code: self.ts.express(
                    ts_code=c,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                )
                df = self._retry(fn, f"express {code}")
                if df is not None:
                    self._save_cache(df, folder, code)
                    if not df.empty:
                        rows.append(df)
                time.sleep(max(self.config.request_pause_sec, 0.12))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        out = out[out["ts_code"].isin(codes)].copy()
        for col in ["ann_date", "end_date"]:
            if col in out.columns:
                out[col] = self._parse_trade_date(out[col])
        out["source_endpoint"] = folder
        out = out.sort_values([c for c in ["ts_code", "end_date", "ann_date"] if c in out.columns]).reset_index(drop=True)
        self.logger.info("Express rows collected: %s", len(out))
        return out

    def get_fina_indicator_data(self, stocks: pd.DataFrame) -> pd.DataFrame:
        if stocks.empty or self.ts is None or not self.config.use_fina_indicator:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []
        codes = list(stocks["ts_code"].dropna().unique())
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, 6)) as ex:
            future_map = {ex.submit(self._get_fina_indicator_single, code): code for code in codes}
            for fut in as_completed(future_map):
                df = fut.result()
                if df is not None and not df.empty:
                    rows.append(df)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        for col in ["ann_date", "end_date"]:
            if col in out.columns:
                out[col] = self._parse_trade_date(out[col])
        out["source_endpoint"] = "fina_indicator"
        out = out.sort_values([c for c in ["ts_code", "end_date", "ann_date"] if c in out.columns]).reset_index(drop=True)
        self.logger.info("Fina indicator rows collected: %s", len(out))
        return out

    def _get_fina_indicator_single(self, ts_code: str) -> pd.DataFrame | None:
        cached = self._load_cache("fina_indicator", ts_code)
        if cached is not None and not cached.empty:
            return cached
        if self.ts is None:
            return None
        try:
            fn = lambda: self.ts.fina_indicator(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            df = self._retry(fn, f"fina_indicator {ts_code}")
            if df is None:
                return None
            if not df.empty:
                self._save_cache(df, "fina_indicator", ts_code)
            return df
        except Exception as exc:
            self.logger.warning("fina_indicator failed for %s: %s", ts_code, exc)
            return None

    def get_report_rc_data(self) -> pd.DataFrame:
        if self.ts is None or not self.config.use_report_rc:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []
        months = pd.period_range(
            start=pd.to_datetime(self.config.start_date, format="%Y%m%d"),
            end=pd.to_datetime(self.config.end_date, format="%Y%m%d"),
            freq="M",
        )
        for month in months:
            start = month.start_time.strftime("%Y%m%d")
            end = month.end_time.strftime("%Y%m%d")
            cache_key = f"{start}_{end}"
            cached = self._load_cache("report_rc", cache_key)
            if cached is not None:
                if not cached.empty:
                    rows.append(cached)
                continue
            fn = lambda s=start, e=end: self.ts.report_rc(start_date=s, end_date=e)
            df = self._retry(fn, f"report_rc {cache_key}")
            if df is not None:
                self._save_cache(df, "report_rc", cache_key)
                if not df.empty:
                    rows.append(df)
            time.sleep(max(self.config.request_pause_sec, 0.12))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        if "report_date" in out.columns:
            out["report_date"] = self._parse_trade_date(out["report_date"])
        out["source_endpoint"] = "report_rc"
        out = out.sort_values([c for c in ["ts_code", "quarter", "report_date"] if c in out.columns]).reset_index(drop=True)
        self.logger.info("report_rc rows collected: %s", len(out))
        return out

    def get_stock_prices(self, stocks: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        if stocks.empty:
            return pd.DataFrame()
        jobs = [(row["ts_code"], row["symbol"]) for _, row in stocks.iterrows()]
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, 8)) as ex:
            future_map = {
                ex.submit(self._get_single_stock_price, ts_code, symbol): ts_code
                for ts_code, symbol in jobs
            }
            for fut in as_completed(future_map):
                price_df = fut.result()
                if price_df is not None and not price_df.empty:
                    rows.append(price_df)
        if not rows:
            self.logger.warning("No stock price data collected.")
            return pd.DataFrame()
        prices = pd.concat(rows, ignore_index=True)
        prices["trade_date"] = self._parse_trade_date(prices["trade_date"])
        prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        prices["ret"] = pd.to_numeric(prices["close"], errors="coerce").groupby(prices["ts_code"]).pct_change()
        self.logger.info("Stock daily prices collected: %s rows", len(prices))
        return prices

    def _get_single_stock_price(self, ts_code: str, symbol: str) -> pd.DataFrame | None:
        cached = self._load_cache("prices", ts_code)
        if cached is not None and not cached.empty:
            return cached
        if self.ts is not None:
            try:
                fn = lambda: self.ts.daily(
                    ts_code=ts_code,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
                df = self._retry(fn, f"daily {ts_code}")
                if df is None:
                    df = pd.DataFrame()
                if not df.empty:
                    self._save_cache(df, "prices", ts_code)
                    return df
            except Exception as exc:
                self.logger.warning("Tushare daily failed for %s: %s", ts_code, exc)
        if self.ak is not None:
            try:
                df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    adjust="qfq",
                )
                col_map = {
                    "日期": "trade_date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "vol",
                    "成交额": "amount",
                }
                df = df.rename(columns=col_map)
                needed = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
                missing = [c for c in needed if c not in df.columns]
                if missing:
                    return None
                df = df[needed].copy()
                df["ts_code"] = ts_code
                self._save_cache(df, "prices", ts_code)
                return df
            except Exception as exc:
                self.logger.warning("Akshare daily fallback failed for %s: %s", ts_code, exc)
        return None

    def get_market_index(self) -> pd.DataFrame:
        cache_key = f"market_{self.config.market_index_code_tushare}_{self.config.start_date}_{self.config.end_date}"
        cached = self._load_cache("market", cache_key)
        if cached is not None and not cached.empty:
            cached["trade_date"] = self._parse_trade_date(cached["trade_date"])
            if "mkt_ret" not in cached.columns:
                cached = cached.sort_values("trade_date").reset_index(drop=True)
                cached["mkt_ret"] = pd.to_numeric(cached["close"], errors="coerce").pct_change()
            return cached
        if self.ts is not None:
            try:
                fn = lambda: self.ts.index_daily(
                    ts_code=self.config.market_index_code_tushare,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    fields="ts_code,trade_date,close",
                )
                df = self._retry(fn, "index_daily market")
                if df is None:
                    df = pd.DataFrame()
                if not df.empty:
                    df["trade_date"] = self._parse_trade_date(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    df["mkt_ret"] = pd.to_numeric(df["close"], errors="coerce").pct_change()
                    self._save_cache(df, "market", cache_key)
                    self.logger.info("Market index from Tushare: %s rows", len(df))
                    return df
            except Exception as exc:
                self.logger.warning("Tushare index_daily failed: %s", exc)
        if self.ak is not None:
            try:
                df = self.ak.stock_zh_index_daily_em(symbol=self.config.market_index_symbol_akshare)
                df = df.rename(columns={"date": "trade_date", "close": "close"})
                if "trade_date" not in df.columns or "close" not in df.columns:
                    return pd.DataFrame()
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df[(df["trade_date"] >= pd.to_datetime(self.config.start_date)) & (df["trade_date"] <= pd.to_datetime(self.config.end_date))]
                df["ts_code"] = self.config.market_index_code_tushare
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["mkt_ret"] = pd.to_numeric(df["close"], errors="coerce").pct_change()
                self._save_cache(df, "market", cache_key)
                self.logger.info("Market index from Akshare: %s rows", len(df))
                return df
            except Exception as exc:
                self.logger.warning("Akshare market fallback failed: %s", exc)
        self.logger.error("Market index collection failed.")
        return pd.DataFrame(columns=["ts_code", "trade_date", "close", "mkt_ret"])

    def get_daily_basic(self, stocks: pd.DataFrame) -> pd.DataFrame:
        if self.ts is None or stocks.empty:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []
        codes = list(stocks["ts_code"].dropna().unique())
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, 6)) as ex:
            future_map = {ex.submit(self._get_daily_basic_single, ts_code): ts_code for ts_code in codes}
            for fut in as_completed(future_map):
                df = fut.result()
                if df is not None and not df.empty:
                    rows.append(df)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        out["trade_date"] = self._parse_trade_date(out["trade_date"])
        numeric_cols = [
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "total_mv",
            "circ_mv",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.logger.info("Daily basic data collected: %s rows", len(out))
        return out

    def _get_daily_basic_single(self, ts_code: str) -> pd.DataFrame | None:
        cached = self._load_cache("daily_basic", ts_code)
        required_cols = {"turnover_rate", "turnover_rate_f", "ps_ttm", "circ_mv"}
        if cached is not None and not cached.empty and required_cols.issubset(set(cached.columns)):
            return cached
        if self.ts is None:
            return None
        try:
            fn = lambda: self.ts.daily_basic(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                fields=(
                    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
                    "pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv"
                ),
            )
            df = self._retry(fn, f"daily_basic {ts_code}")
            if df is None:
                return None
            if not df.empty:
                self._save_cache(df, "daily_basic", ts_code)
            return df
        except Exception as exc:
            self.logger.warning("daily_basic failed for %s: %s", ts_code, exc)
            return None
