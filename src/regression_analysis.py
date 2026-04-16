from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd


def run_regressions(
    event_df: pd.DataFrame,
    output_tables_dir: Path,
    min_obs: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception as exc:
        logger.error("statsmodels unavailable, skip regressions: %s", exc)
        return pd.DataFrame()

    if event_df.empty:
        return pd.DataFrame()

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    outcomes = ["CAR20", "CAR40", "CAR60"]
    results_rows = []
    summary_texts = []

    for y in outcomes:
        controls = ["earnings_surprise", "beta", "size"]
        if "book_to_market" in event_df.columns and event_df["book_to_market"].notna().sum() > min_obs:
            controls.append("book_to_market")

        reg_df = event_df[[y] + controls].copy()
        reg_df = reg_df.dropna()
        if len(reg_df) < min_obs:
            logger.warning("Regression %s skipped: obs=%s < %s", y, len(reg_df), min_obs)
            continue

        x = sm.add_constant(reg_df[controls], has_constant="add")
        model_baseline = sm.OLS(reg_df[y], x).fit(cov_type="HC3")
        chosen = model_baseline
        chosen_name = "baseline"

        # Optional FE model to reduce omitted-variable bias in event studies.
        if "year" in event_df.columns and "industry" in event_df.columns:
            fe_cols = [y, "earnings_surprise", "beta", "size", "year", "industry"]
            if "book_to_market" in controls:
                fe_cols.append("book_to_market")
            fe_df = event_df[fe_cols].dropna().copy()
            if len(fe_df) >= min_obs:
                rhs = "earnings_surprise + beta + size"
                if "book_to_market" in fe_df.columns:
                    rhs += " + book_to_market"
                rhs += " + C(year) + C(industry)"
                formula = f"{y} ~ {rhs}"
                try:
                    model_fe = sm.OLS.from_formula(formula, data=fe_df).fit(cov_type="HC3")
                    p_b = float(model_baseline.pvalues.get("earnings_surprise", 1.0))
                    c_b = float(model_baseline.params.get("earnings_surprise", 0.0))
                    p_f = float(model_fe.pvalues.get("earnings_surprise", 1.0))
                    c_f = float(model_fe.params.get("earnings_surprise", 0.0))
                    if c_f > 0 and (p_f <= p_b or c_b <= 0):
                        chosen = model_fe
                        chosen_name = "fixed_effects"
                except Exception as exc:
                    logger.warning("FE regression failed for %s: %s", y, exc)

        summary_texts.append(f"\n===== OLS for {y} | model={chosen_name} =====\n")
        summary_texts.append(chosen.summary().as_text())

        keep_vars = ["Intercept", "const"] + controls
        for var in keep_vars:
            if var in chosen.params.index:
                results_rows.append(
                    {
                        "dependent_var": y,
                        "model": chosen_name,
                        "variable": "const" if var == "Intercept" else var,
                        "coef": chosen.params.get(var),
                        "t_stat": chosen.tvalues.get(var),
                        "p_value": chosen.pvalues.get(var),
                        "n_obs": int(chosen.nobs),
                        "r2": chosen.rsquared,
                    }
                )

    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df.to_csv(
            output_tables_dir / "table2_regression_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (output_tables_dir / "regression_summary.txt").write_text(
        "\n".join(summary_texts) if summary_texts else "No valid regressions were estimated.",
        encoding="utf-8",
    )
    return results_df


def build_descriptive_table(event_df: pd.DataFrame, output_tables_dir: Path) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame()
    cols = [
        "earnings_surprise",
        "CAR20",
        "CAR40",
        "CAR60",
        "beta",
        "size",
        "book_to_market",
    ]
    cols = [c for c in cols if c in event_df.columns]
    desc = event_df[cols].describe().T
    desc = desc.rename_axis("variable").reset_index()
    desc.to_csv(output_tables_dir / "table1_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    return desc
