from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 11


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_es_distribution(event_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(event_df["earnings_surprise"].dropna(), bins=40, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.set_title("Distribution of Earnings Surprise")
    ax.set_xlabel("Earnings Surprise")
    ax.set_ylabel("Frequency")
    _save_fig(fig, fig_dir / "fig1_es_distribution.png")


def plot_scatter(event_df: pd.DataFrame, y_col: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = event_df["earnings_surprise"]
    y = event_df[y_col]
    ax.scatter(x, y, alpha=0.6, color="#1F77B4", edgecolors="none")
    if x.notna().sum() > 2 and y.notna().sum() > 2:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 2:
            coef = np.polyfit(x[m], y[m], 1)
            x_line = np.linspace(x[m].min(), x[m].max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="black", linewidth=1.2)
    ax.set_title(f"Earnings Surprise vs {y_col}")
    ax.set_xlabel("Earnings Surprise")
    ax.set_ylabel(y_col)
    _save_fig(fig, Path(filename))


def plot_quantile_group_bars(event_df: pd.DataFrame, fig_dir: Path, n_groups: int = 5) -> None:
    df = event_df.copy()
    df = df.dropna(subset=["earnings_surprise", "CAR20", "CAR40", "CAR60"])
    df["es_group"] = pd.qcut(df["earnings_surprise"], q=n_groups, labels=False, duplicates="drop") + 1
    g = df.groupby("es_group")[["CAR20", "CAR40", "CAR60"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(g))
    width = 0.25
    ax.bar(x - width, g["CAR20"], width, label="CAR20")
    ax.bar(x, g["CAR40"], width, label="CAR40")
    ax.bar(x + width, g["CAR60"], width, label="CAR60")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{int(i)}" for i in g["es_group"]])
    ax.set_title("Average CAR by Earnings Surprise Quantiles")
    ax.set_xlabel("Earnings Surprise Quantile Group")
    ax.set_ylabel("Average CAR")
    ax.legend(frameon=False)
    _save_fig(fig, fig_dir / "fig4_quantile_group_car_comparison.png")


def plot_high_low_caar(path_df: pd.DataFrame, fig_dir: Path, n_groups: int = 5) -> None:
    if path_df.empty:
        return
    df = path_df.copy().dropna(subset=["earnings_surprise", "event_day", "abret"])
    df["es_group"] = pd.qcut(df["earnings_surprise"], q=n_groups, labels=False, duplicates="drop") + 1
    if df["es_group"].nunique() < 2:
        return

    low = df["es_group"].min()
    high = df["es_group"].max()
    low_curve = (
        df[df["es_group"] == low].groupby("event_day")["abret"].mean().sort_index().cumsum()
    )
    high_curve = (
        df[df["es_group"] == high].groupby("event_day")["abret"].mean().sort_index().cumsum()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(high_curve.index, high_curve.values, label="High ES Group", linewidth=2.0)
    ax.plot(low_curve.index, low_curve.values, label="Low ES Group", linewidth=2.0)
    ax.axhline(0, color="gray", linewidth=1.0)
    ax.set_title("Post-Announcement Cumulative Abnormal Return")
    ax.set_xlabel("Event Day")
    ax.set_ylabel("CAAR")
    ax.legend(frameon=False)
    _save_fig(fig, fig_dir / "fig5_high_vs_low_es_caar.png")


def plot_regression_coef(reg_df: pd.DataFrame, fig_dir: Path) -> None:
    if reg_df.empty:
        return
    d = reg_df[reg_df["variable"] == "earnings_surprise"].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(d["dependent_var"], d["coef"], color="#2C7FB8")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("Earnings Surprise Coefficient Across CAR Regressions")
    ax.set_xlabel("Dependent Variable")
    ax.set_ylabel("Coefficient")
    _save_fig(fig, fig_dir / "fig6_regression_es_coefficient.png")


def generate_all_figures(
    event_df: pd.DataFrame,
    path_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    fig_dir: Path,
    n_groups: int = 5,
) -> None:
    if event_df.empty:
        return
    plot_es_distribution(event_df, fig_dir)
    plot_scatter(event_df, "CAR20", str(fig_dir / "fig2_es_vs_car20.png"))
    plot_scatter(event_df, "CAR60", str(fig_dir / "fig3_es_vs_car60.png"))
    plot_quantile_group_bars(event_df, fig_dir, n_groups=n_groups)
    plot_high_low_caar(path_df, fig_dir, n_groups=n_groups)
    plot_regression_coef(reg_df, fig_dir)
