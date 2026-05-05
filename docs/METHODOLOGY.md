# Methodology

This document details the empirical design and measurement choices for the earnings surprise diagnostics framework.

## 1. Event Definition
We categorize earnings-related announcements into four standardized event types:
- **Preannouncement (`preannouncement`)**: Initial management earnings guidance.
- **Revision (`revision`)**: Updates to previously issued management guidance.
- **Express (`express`)**: Preliminary earnings results (业绩快报).
- **Formal Release (`formal_release`)**: Final audited financial statements.

The default headline baseline uses only **Preannouncements** to minimize information leakage from prior releases.

## 2. Surprise Definitions
We compute three parallel surprise metrics:
- **Raw Surprise (`main_surprise_raw`)**: $Actual - Expected$
- **Percentage Surprise (`main_surprise_pct`)**: $(Actual - Expected) / |Expected|$
- **Standardized Surprise (`main_surprise_std`)**: $Winsorized\_Pct\_Surprise / \sigma(Winsorized\_Pct\_Surprise)$

All surprise metrics are winsorized at the 1% and 99% levels to handle outliers in financial reporting data.

## 3. Abnormal Return (AR) Models
- **Market-Adjusted (Baseline)**: $AR_{it} = R_{it} - R_{mt}$, where $R_{mt}$ is the CSI 300 index return.
- **Industry-Adjusted**: $AR_{it} = R_{it} - R_{industry,t}$, where $R_{industry,t}$ is the equal-weighted mean return of all sampled stocks in the same CSRC industry on day $t$. Falls back to market return when industry data is unavailable. Produces `IAR{w}` columns alongside `CAR{w}`.
- **Market Model Residual (Hook)**: $AR_{it} = R_{it} - (\hat{\alpha}_i + \hat{\beta}_i R_{mt})$.

## 4. Event Windows (CAR)
- **Pre-event leakage**: CAR[-10,-1], CAR[-5,-1]
- **Immediate reaction**: CAR[0,1], CAR[0,3], CAR[0,5]
- **Post-event drift**: CAR[1,10], CAR[1,20], CAR[1,60]

The event day (day 0) is the announcement date or the first following trading day if the announcement occurs on a non-trading day.

## 5. Matching Hierarchy
Expectations are matched to events using a strict-to-relaxed hierarchy:
1. **Strict Same Quarter**: Expectation period matches event period exactly.
2. **Same Fiscal Year Nearest Valid**: Matches nearest available forecast for the same fiscal year.
3. **Latest Valid Pre-Event**: Uses the most recent forecast regardless of period (Legacy fallback).

## 6. Regression Specification
$CAR_{i,w} = \alpha + \gamma \cdot Surprise_{i} + \mathbf{X}_{i}'\beta + \delta_{ind} + \delta_{qtr} + \epsilon_{i}$
- **Controls ($X_i$)**: Size (log MV), Beta, Book-to-Market, Turnover, PE, PS.
- **Fixed Effects**: Industry ($\delta_{ind}$), Year-Quarter ($\delta_{qtr}$).
- **Clustering**: Standard errors are clustered at the firm level.

## 7. Limitations
- **Analyst Bias**: Expectations are subject to stale forecasts and analyst optimism.
- **Timing**: Announcements after market close are mapped to $T+1$ but intraday timing is often noisy.
- **Market Model**: Simple market-adjusted returns may not account for risk-factor exposures fully.
