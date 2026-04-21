from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.io_utils import save_csv, save_text


def save_tushare_outputs(
    event_df: pd.DataFrame,
    path_df: pd.DataFrame,
    outputs_tables_dir: Path,
    outputs_audit_dir: Path,
    scenario_name: str = "tushare_first",
) -> dict[str, float | str]:
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_audit_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = outputs_tables_dir / f"event_dataset_{scenario_name}.csv"
    reg_path = outputs_tables_dir / f"regression_results_{scenario_name}.csv"
    event_type_path = outputs_tables_dir / f"event_counts_by_type_{scenario_name}.csv"
    year_path = outputs_tables_dir / f"event_counts_by_year_{scenario_name}.csv"
    coverage_path = outputs_audit_dir / f"signal_coverage_by_year_{scenario_name}.csv"
    missing_path = outputs_audit_dir / f"missingness_summary_{scenario_name}.csv"
    audit_note_path = outputs_audit_dir / f"audit_note_{scenario_name}.txt"

    save_csv(event_df, dataset_path)
    save_csv(path_df, outputs_tables_dir / f"event_paths_{scenario_name}.csv")

    if event_df.empty:
        save_csv(pd.DataFrame(), reg_path)
        save_csv(pd.DataFrame(), event_type_path)
        save_csv(pd.DataFrame(), year_path)
        save_csv(pd.DataFrame(), coverage_path)
        save_csv(pd.DataFrame(), missing_path)
        save_text("No valid events after filters.\n", audit_note_path)
        return {
            "scenario": scenario_name,
            "sample_size": 0,
            "headline_window": "CAR5",
            "headline_coef": np.nan,
            "headline_p_value": np.nan,
        }

    df = event_df.copy()
    df["event_trade_date"] = pd.to_datetime(df["event_trade_date"], errors="coerce")
    df["year"] = df["event_trade_date"].dt.year
    df["year_quarter"] = df["event_trade_date"].dt.to_period("Q").astype(str)
    df["industry"] = df["industry"].fillna("unknown")
    df["event_type"] = df["event_type"].fillna("unknown")

    event_counts_by_type = df.groupby("event_type", as_index=False).size().rename(columns={"size": "event_count"})
    event_counts_by_year = df.groupby("year", as_index=False).size().rename(columns={"size": "event_count"})
    coverage = (
        df.groupby("year", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            strict_report_rc_count=("report_rc_match_quality", lambda s: int((s == "strict").sum())),
            usable_signal_count=("main_surprise_std", lambda s: int(s.notna().sum())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    coverage["strict_report_rc_share"] = coverage["strict_report_rc_count"] / coverage["event_count"].replace(0, np.nan)
    coverage["usable_signal_share"] = coverage["usable_signal_count"] / coverage["event_count"].replace(0, np.nan)

    missing_cols = [
        "main_surprise_std",
        "expected_np",
        "expected_eps",
        "turnover20",
        "beta",
        "total_mv",
        "pb",
        "ps_ttm",
    ] + [c for c in df.columns if c.startswith("CAR")]
    missingness = pd.DataFrame(
        {
            "column": missing_cols,
            "missing_count": [int(df[c].isna().sum()) if c in df.columns else len(df) for c in missing_cols],
            "missing_share": [float(df[c].isna().mean()) if c in df.columns else 1.0 for c in missing_cols],
        }
    )

    reg = run_tushare_regressions(df)

    save_csv(event_counts_by_type, event_type_path)
    save_csv(event_counts_by_year, year_path)
    save_csv(coverage, coverage_path)
    save_csv(missingness, missing_path)
    save_csv(reg, reg_path)

    note_lines = [
        f"Tushare-first audit note ({scenario_name})",
        f"Sample size: {len(df)}",
        f"Event types: {', '.join(sorted(df['event_type'].dropna().unique()))}",
        f"Strict report_rc matches: {(df['report_rc_match_quality'] == 'strict').sum()} / {len(df)}",
        f"Usable standardized surprise rows: {df['main_surprise_std'].notna().sum()} / {len(df)}",
        "Headline specification: CAR5 on standardized surprise with controls, industry FE, year-quarter FE, event-type FE, and firm-clustered SE.",
        "Interpret the preferred result as a cleaner short-window valuation-adjustment design; treat fallback or weak-match samples as supplementary.",
    ]
    save_text("\n".join(note_lines) + "\n", audit_note_path)

    headline = reg[(reg["dependent_var"] == "CAR5") & (reg["model"] == "pooled_fe") & (reg["variable"] == "main_surprise_std")]
    return {
        "scenario": scenario_name,
        "sample_size": int(len(df)),
        "headline_window": "CAR5",
        "headline_coef": float(headline["coef"].iloc[0]) if not headline.empty else np.nan,
        "headline_p_value": float(headline["p_value"].iloc[0]) if not headline.empty else np.nan,
    }


def run_tushare_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:
        return pd.DataFrame()

    car_windows = [c for c in df.columns if c.startswith("CAR")]
    for dep in car_windows:
        regressors = [
            "main_surprise_std",
            "log_total_mv",
            "beta",
            "book_to_market",
            "turnover20",
            "pe_ttm",
            "ps_ttm",
        ]
        cols = [dep, "ts_code", "industry", "year_quarter", "event_type"] + regressors
        dd = df[cols].copy()
        for col in [dep] + regressors:
            if col in dd.columns:
                dd[col] = pd.to_numeric(dd[col], errors="coerce")
                dd[col] = dd[col].replace([np.inf, -np.inf], np.nan)
        dd = dd.dropna(subset=[dep, "main_surprise_std", "ts_code", "industry", "year_quarter", "event_type"]).copy()
        if len(dd) >= 40:
            active_regs = [r for r in regressors if r in dd.columns and dd[r].notna().sum() >= 30]
            reg_dd = dd[[dep, "ts_code", "industry", "year_quarter", "event_type"] + active_regs].dropna().copy()
            if len(reg_dd) >= 40 and active_regs:
                formula = f"{dep} ~ {' + '.join(active_regs)} + C(industry) + C(year_quarter) + C(event_type)"
                model = smf.ols(formula=formula, data=reg_dd).fit(cov_type="cluster", cov_kwds={"groups": reg_dd["ts_code"]})
                for var in active_regs:
                    rows.append(
                        {
                            "model": "pooled_fe",
                            "dependent_var": dep,
                            "variable": var,
                            "coef": model.params.get(var, np.nan),
                            "t_stat": model.tvalues.get(var, np.nan),
                            "p_value": model.pvalues.get(var, np.nan),
                            "n_obs": int(model.nobs),
                            "r2": model.rsquared,
                        }
                    )
            else:
                rows.append(
                    {
                        "model": "pooled_fe",
                        "dependent_var": dep,
                        "variable": "insufficient_obs",
                        "coef": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(reg_dd)),
                        "r2": np.nan,
                    }
                )
        else:
            rows.append(
                {
                    "model": "pooled_fe",
                    "dependent_var": dep,
                    "variable": "insufficient_obs",
                    "coef": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(dd)),
                    "r2": np.nan,
                }
            )

        for event_type, subset in df.groupby("event_type"):
            dsub = subset[cols].copy()
            for col in [dep] + regressors:
                if col in dsub.columns:
                    dsub[col] = pd.to_numeric(dsub[col], errors="coerce")
                    dsub[col] = dsub[col].replace([np.inf, -np.inf], np.nan)
            dsub = dsub.dropna(subset=[dep, "main_surprise_std", "ts_code", "industry", "year_quarter"]).copy()
            if len(dsub) < 30:
                rows.append(
                    {
                        "model": f"event_type_{event_type}",
                        "dependent_var": dep,
                        "variable": "insufficient_obs",
                        "coef": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(dsub)),
                        "r2": np.nan,
                    }
                )
                continue
            active_regs = [r for r in regressors if r in dsub.columns and dsub[r].notna().sum() >= 20]
            reg_dsub = dsub[[dep, "ts_code", "industry", "year_quarter"] + active_regs].dropna().copy()
            if len(reg_dsub) < 30 or not active_regs:
                rows.append(
                    {
                        "model": f"event_type_{event_type}",
                        "dependent_var": dep,
                        "variable": "insufficient_obs",
                        "coef": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(reg_dsub)),
                        "r2": np.nan,
                    }
                )
                continue
            formula = f"{dep} ~ {' + '.join(active_regs)} + C(industry) + C(year_quarter)"
            model = smf.ols(formula=formula, data=reg_dsub).fit(cov_type="cluster", cov_kwds={"groups": reg_dsub["ts_code"]})
            for var in active_regs:
                rows.append(
                    {
                        "model": f"event_type_{event_type}",
                        "dependent_var": dep,
                        "variable": var,
                        "coef": model.params.get(var, np.nan),
                        "t_stat": model.tvalues.get(var, np.nan),
                        "p_value": model.pvalues.get(var, np.nan),
                        "n_obs": int(model.nobs),
                        "r2": model.rsquared,
                    }
                )
    return pd.DataFrame(rows)
