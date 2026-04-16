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
    earnings: pd.DataFrame
    prices: pd.DataFrame
    market: pd.DataFrame
    daily_basic: pd.DataFrame
    cross_check: pd.DataFrame


class DataCollector:
    """Collect A-share data with Tushare primary and Akshare fallback."""

    def __init__(self, config: ProjectConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.ts = None
        self.ak = None
        self._load_clients()
        self.cache_root = self.config.data_raw_dir / "cache"
        for folder in ["earnings", "guidance", "prices", "daily_basic", "market", "cross_check"]:
            (self.cache_root / folder).mkdir(parents=True, exist_ok=True)

    def _load_clients(self) -> None:
        try:
            import tushare as ts  # type: ignore

            self.ts = ts.pro_api(self.config.tushare_token)
            self.logger.info("Tushare client initialized.")
        except Exception as exc:
            self.logger.warning("Tushare unavailable: %s", exc)

        try:
            import akshare as ak  # type: ignore

            self.ak = ak
            self.logger.info("Akshare client initialized.")
        except Exception as exc:
            self.logger.warning("Akshare unavailable: %s", exc)

    def collect_all(self) -> DataBundle:
        stocks = self.get_stock_universe()
        earnings = self.get_earnings_data(stocks)
        prices = self.get_stock_prices(stocks)
        market = self.get_market_index()
        daily_basic = self.get_daily_basic(stocks)
        cross_check = self.build_cross_check(stocks) if self.config.enable_cross_check else pd.DataFrame()

        save_csv(stocks, self.config.data_raw_dir / "stock_universe.csv")
        save_csv(earnings, self.config.data_raw_dir / "earnings_raw.csv")
        save_csv(prices, self.config.data_raw_dir / "stock_prices_raw.csv")
        save_csv(market, self.config.data_raw_dir / "market_index_raw.csv")
        save_csv(daily_basic, self.config.data_raw_dir / "daily_basic_raw.csv")
        save_csv(cross_check, self.config.data_raw_dir / "cross_check_price_sources.csv")

        return DataBundle(
            stocks=stocks,
            earnings=earnings,
            prices=prices,
            market=market,
            daily_basic=daily_basic,
            cross_check=cross_check,
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

    def get_stock_universe(self) -> pd.DataFrame:
        # Prefer Tushare's stock_basic for structured fields (industry, list date).
        if self.ts is not None:
            try:
                df = self.ts.stock_basic(
                    exchange="",
                    list_status="L",
                    fields="ts_code,symbol,name,area,industry,list_date",
                )
                if not df.empty:
                    df = df.sort_values("ts_code").reset_index(drop=True)
                    if self.config.sample_stock_count:
                        df = df.head(self.config.sample_stock_count).copy()
                    self.logger.info("Stock universe from Tushare: %s rows", len(df))
                    return df
            except Exception as exc:
                self.logger.warning("Tushare stock_basic failed: %s", exc)

        # Fallback to Akshare if Tushare is unavailable.
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

    def get_earnings_data(self, stocks: pd.DataFrame) -> pd.DataFrame:
        all_rows: list[pd.DataFrame] = []
        if stocks.empty:
            return pd.DataFrame()

        codes = list(stocks["ts_code"].dropna().unique())
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            future_map = {ex.submit(self._get_earnings_single_stock, code): code for code in codes}
            for fut in as_completed(future_map):
                df = fut.result()
                if df is not None and not df.empty:
                    all_rows.append(df)

        if not all_rows:
            self.logger.warning("No earnings records collected.")
            return pd.DataFrame()

        earnings = pd.concat(all_rows, ignore_index=True)
        earnings["ann_date"] = pd.to_datetime(earnings["ann_date"], format="%Y%m%d", errors="coerce")
        earnings["end_date"] = pd.to_datetime(earnings["end_date"], format="%Y%m%d", errors="coerce")
        earnings = earnings.dropna(subset=["ts_code", "ann_date", "end_date"])
        earnings = earnings.sort_values(["ts_code", "ann_date", "end_date"]).reset_index(drop=True)
        self.logger.info("Earnings records collected: %s rows", len(earnings))
        return earnings

    def get_guidance_data(self, stocks: pd.DataFrame) -> pd.DataFrame:
        """Collect earnings guidance (业绩预告) from Tushare forecast endpoint."""
        if stocks.empty or self.ts is None:
            return pd.DataFrame()

        rows: list[pd.DataFrame] = []
        codes = list(stocks["ts_code"].dropna().unique())
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            future_map = {ex.submit(self._get_guidance_single_stock, code): code for code in codes}
            for fut in as_completed(future_map):
                df = fut.result()
                if df is not None and not df.empty:
                    rows.append(df)

        if not rows:
            self.logger.warning("No guidance rows collected.")
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)
        out["ann_date"] = pd.to_datetime(out["ann_date"], format="%Y%m%d", errors="coerce")
        out["end_date"] = pd.to_datetime(out["end_date"], format="%Y%m%d", errors="coerce")
        if "first_ann_date" in out.columns:
            out["first_ann_date"] = pd.to_datetime(out["first_ann_date"], format="%Y%m%d", errors="coerce")
        out = out.dropna(subset=["ts_code", "ann_date", "end_date"])
        out = out.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        self.logger.info("Guidance rows collected: %s", len(out))
        return out

    def _get_guidance_single_stock(self, ts_code: str) -> pd.DataFrame | None:
        cached = self._load_cache("guidance", ts_code)
        if cached is not None and not cached.empty:
            return cached
        if self.ts is None:
            return None
        try:
            fn = lambda: self.ts.forecast(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            df = self._retry(fn, f"forecast {ts_code}")
            if df is None:
                return None
            if not df.empty:
                self._save_cache(df, "guidance", ts_code)
            return df
        except Exception as exc:
            self.logger.warning("forecast failed for %s: %s", ts_code, exc)
            return None

    def _get_earnings_single_stock(self, ts_code: str) -> pd.DataFrame | None:
        cached = self._load_cache("earnings", ts_code)
        if cached is not None and not cached.empty:
            return cached

        if self.ts is not None:
            # Primary approach: use income statement quarterly records.
            for endpoint in ("income", "fina_indicator"):
                try:
                    if endpoint == "income":
                        fn = lambda: self.ts.income(
                            ts_code=ts_code,
                            start_date=self.config.start_date,
                            end_date=self.config.end_date,
                            fields=(
                                "ts_code,ann_date,end_date,"
                                "n_income_attr_p,basic_eps,diluted_eps"
                            ),
                        )
                        df = self._retry(fn, f"income {ts_code}")
                        if df is None:
                            continue
                        if not df.empty:
                            df["actual_earnings"] = (
                                df["n_income_attr_p"]
                                .fillna(df["basic_eps"])
                                .fillna(df["diluted_eps"])
                            )
                            out = df[
                                [
                                    "ts_code",
                                    "ann_date",
                                    "end_date",
                                    "n_income_attr_p",
                                    "basic_eps",
                                    "diluted_eps",
                                    "actual_earnings",
                                ]
                            ]
                            self._save_cache(out, "earnings", ts_code)
                            return out
                    else:
                        fn = lambda: self.ts.fina_indicator(
                            ts_code=ts_code,
                            start_date=self.config.start_date,
                            end_date=self.config.end_date,
                            fields="ts_code,ann_date,end_date,eps,dt_eps",
                        )
                        df = self._retry(fn, f"fina_indicator {ts_code}")
                        if df is None:
                            continue
                        if not df.empty:
                            df["actual_earnings"] = df["dt_eps"].fillna(df["eps"])
                            df["n_income_attr_p"] = np.nan
                            df["basic_eps"] = df["eps"]
                            df["diluted_eps"] = np.nan
                            out = df[
                                [
                                    "ts_code",
                                    "ann_date",
                                    "end_date",
                                    "n_income_attr_p",
                                    "basic_eps",
                                    "diluted_eps",
                                    "actual_earnings",
                                ]
                            ]
                            self._save_cache(out, "earnings", ts_code)
                            return out
                except Exception as exc:
                    self.logger.warning(
                        "Tushare %s failed for %s: %s", endpoint, ts_code, exc
                    )

        # Akshare fallback for financials is less standardized across symbols/endpoints.
        # We return None if unavailable and allow downstream logic to drop sparse records.
        return None

    def get_stock_prices(self, stocks: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        if stocks.empty:
            return pd.DataFrame()

        jobs = [(row["ts_code"], row["symbol"]) for _, row in stocks.iterrows()]
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
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
        prices["ret"] = prices.groupby("ts_code")["close"].pct_change()
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
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    df["mkt_ret"] = df["close"].pct_change()
                    self._save_cache(df, "market", cache_key)
                    self.logger.info("Market index from Tushare: %s rows", len(df))
                    return df
            except Exception as exc:
                self.logger.warning("Tushare index_daily failed: %s", exc)

        if self.ak is not None:
            # Akshare fallback using index daily close.
            try:
                df = self.ak.stock_zh_index_daily_em(symbol=self.config.market_index_symbol_akshare)
                col_map = {"date": "trade_date", "close": "close"}
                df = df.rename(columns=col_map)
                if "trade_date" not in df.columns or "close" not in df.columns:
                    return pd.DataFrame()
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df[(df["trade_date"] >= pd.to_datetime(self.config.start_date)) &
                        (df["trade_date"] <= pd.to_datetime(self.config.end_date))]
                df["ts_code"] = self.config.market_index_code_tushare
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["mkt_ret"] = df["close"].pct_change()
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
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            future_map = {ex.submit(self._get_daily_basic_single, ts_code): ts_code for ts_code in codes}
            for fut in as_completed(future_map):
                df = fut.result()
                if df is not None and not df.empty:
                    rows.append(df)

        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.logger.info("Daily basic data collected: %s rows", len(out))
        return out

    def _get_daily_basic_single(self, ts_code: str) -> pd.DataFrame | None:
        cached = self._load_cache("daily_basic", ts_code)
        if cached is not None and not cached.empty:
            return cached
        try:
            fn = lambda: self.ts.daily_basic(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                fields="ts_code,trade_date,total_mv,pb,pe_ttm",
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

    def build_cross_check(self, stocks: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-check key fields between Tushare and Akshare.
        Here we compare daily close prices for a few sample stocks.
        """
        if self.ts is None or self.ak is None or stocks.empty:
            return pd.DataFrame()

        check_rows: list[pd.DataFrame] = []
        check_sample = stocks.head(min(self.config.cross_check_sample_count, len(stocks)))

        for _, row in check_sample.iterrows():
            ts_code = row["ts_code"]
            symbol = row["symbol"]
            try:
                ts_df = self.ts.daily(
                    ts_code=ts_code,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    fields="trade_date,close",
                )
                ak_df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    adjust="qfq",
                )
                ak_df = ak_df.rename(columns={"日期": "trade_date", "收盘": "close_ak"})
                if ts_df.empty or ak_df.empty or "close_ak" not in ak_df.columns:
                    continue
                ts_df["trade_date"] = pd.to_datetime(ts_df["trade_date"])
                ak_df["trade_date"] = pd.to_datetime(ak_df["trade_date"])
                merged = ts_df.merge(ak_df[["trade_date", "close_ak"]], on="trade_date", how="inner")
                if merged.empty:
                    continue
                merged["ts_code"] = ts_code
                merged["abs_diff"] = (merged["close"] - merged["close_ak"]).abs()
                merged["pct_diff"] = merged["abs_diff"] / merged["close"].abs().replace(0, np.nan)
                check_rows.append(merged[["ts_code", "trade_date", "close", "close_ak", "abs_diff", "pct_diff"]])
            except Exception as exc:
                self.logger.warning("Cross-check failed for %s: %s", ts_code, exc)

        if not check_rows:
            return pd.DataFrame()

        check_df = pd.concat(check_rows, ignore_index=True)
        self.logger.info("Cross-check rows: %s", len(check_df))
        return check_df
