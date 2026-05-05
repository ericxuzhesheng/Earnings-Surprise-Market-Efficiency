from pathlib import Path
import re

import pandas as pd


def _metric_map(df: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["metric"]): str(row["value"])
        for _, row in df.iterrows()
        if pd.notna(row.get("metric"))
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _display_match_tier(raw_value: str) -> str:
    mapping = {
        "strict_same_quarter": "Strict Same Quarter",
        "same_fiscal_year_nearest_valid": "Same Fiscal Year Nearest Valid",
        "latest_valid_pre_event": "Latest Valid Pre-Event",
        "multi_report_median": "Multi-Report Median",
    }
    return mapping.get(raw_value, raw_value.replace("_", " ").title())


def update_readme_results() -> None:
    print("Updating README results from generated CSVs...")

    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"
    tables_dir = project_root / "outputs" / "tables"

    run_summary_path = tables_dir / "run_summary.csv"
    car_summary_path = tables_dir / "event_window_car_summary.csv"
    drift_summary_path = tables_dir / "leakage_drift_diagnostics.csv"

    required = [run_summary_path, car_summary_path, drift_summary_path]
    if not all(path.exists() for path in required):
        missing = [p.name for p in required if not p.exists()]
        print(f"Required CSVs not found ({', '.join(missing)}); skipping README update.")
        return

    run_summary = pd.read_csv(run_summary_path)
    car_summary = pd.read_csv(car_summary_path, index_col=0)
    drift_summary = pd.read_csv(drift_summary_path, index_col=0)
    metrics = _metric_map(run_summary)

    usable_signal_rows = int(float(metrics.get("usable_signal_rows", metrics.get("strict_match_rows", "0"))))
    headline_match_tier = _display_match_tier(metrics.get("headline_match_tier", "strict_same_quarter"))
    car_1_10 = float(car_summary.loc["CAR_1_10", "mean_car"]) if "CAR_1_10" in car_summary.index else float("nan")
    leakage_car = float(drift_summary.loc["CAR_-10_-1", "mean_car"]) if "CAR_-10_-1" in drift_summary.index else float("nan")
    drift_car = float(drift_summary.loc["CAR_1_20", "mean_car"]) if "CAR_1_20" in drift_summary.index else float("nan")

    cn_rows = [
        f"| 基准样本量 (Headline Sample Size) | {usable_signal_rows} (可用信号行) |",
        "| 主事件类型 (Main Event Type) | 业绩预告 (Preannouncement) |",
        f"| 匹配层级 (Matching Tier) | {headline_match_tier} |",
        f"| CAR[1, 10] 均值 (N={usable_signal_rows}) | {_format_pct(car_1_10)} (诊断性证据) |",
        "| 泄露/漂移结论 | 样本量过小，结论不具有统计显著性 (详见 RESULTS_AUDIT.md) |",
    ]

    en_rows = [
        f"| Final Headline Sample Size | {usable_signal_rows} (usable signal rows) |",
        "| Main Event Type | Preannouncement |",
        f"| Matching Tier | {headline_match_tier} |",
        f"| CAR[1, 10] Mean (N={usable_signal_rows}) | {_format_pct(car_1_10)} (diagnostic evidence) |",
        "| Leakage/Drift Conclusion | Inconclusive due to small N (See RESULTS_AUDIT.md) |",
    ]

    content = readme_path.read_text(encoding="utf-8")

    content, cn_replacements = re.subn(
        r"### 关键结果摘要 \(Key Results Snapshot\)\n.*?(?=\n### |\n---|\Z)",
        "### 关键结果摘要 (Key Results Snapshot)\n| 指标 | 结果 |\n| :--- | :--- |\n" + "\n".join(cn_rows) + "\n\n",
        content,
        flags=re.DOTALL,
    )
    content, en_replacements = re.subn(
        r"### Key Results Snapshot\n.*?(?=\n### |\n---|\Z)",
        "### Key Results Snapshot\n| Metric | Value |\n| :--- | :--- |\n" + "\n".join(en_rows) + "\n\n",
        content,
        flags=re.DOTALL,
    )

    if cn_replacements == 0 or en_replacements == 0:
        raise RuntimeError("README snapshot block was not found.")

    with readme_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    print(
        "README.md updated: "
        f"sample_size={usable_signal_rows}, match_tier={headline_match_tier}, "
        f"car_1_10={_format_pct(car_1_10)}, leakage={_format_pct(leakage_car)}, drift20={_format_pct(drift_car)}"
    )


if __name__ == "__main__":
    update_readme_results()
