# Results Audit (2026-05-03)

## Validation Status: Completed (Diagnostic Run)
- **Sample Size**: 300 stocks, 11,413 initial events.
- **Tushare Inputs**: Cached Tushare inputs were used for recovery.
- **Run Mode**: `test` (300 stocks requested).

## Key Audit Findings

### 1. Sample Construction & Matching Quality
- **Funnel Performance**:
    - Initial Events: 11,413
    - Valid Price/Liquidity: 10,599
    - Expectation Match (Any Tier): 1,280
    - Headline Sample (Preannouncement): 2,455
- **Matching Quality**: 
    - The strict match share for preannouncements is approximately 12.5%.
    - Analyst reports used for benchmarks have a median lag of 61 days relative to the event.

### 2. Event-Study Results (N=326 Usable Headline Rows)
- **Immediate Reaction**: CAR[0, 1] is +0.27%.
- **Pre-event Leakage**: CAR[-10, -1] is -0.45%, and CAR[-5, -1] is +0.18%.
- **Post-event Drift**: 
    - CAR[1, 10]: +0.30%
    - CAR_1_20: +2.76%
    - CAR_1_60: +2.68%
- **Interpretation**: The drift pattern is mild and does not survive as a robust headline conclusion.

### 3. Regression & Robustness
- **Regression N**: 245 for the strongest surviving spec (`spec_194`).
- **Strongest Spec**: `latest_per_analyst`, `preannouncement_only`, `raw`, `strict_same_quarter`, `CAR10`.
- **Sensitivity**: The headline coefficient is not statistically strong (`p=0.540583`), so the evidence remains diagnostic rather than decisive.

## Diagnostic Summary
The pipeline is functionally robust and successfully handles the cached Tushare-backed recovery run. The current headline evidence is still diagnostic, not definitive market-efficiency proof.

## Main Limitations
- **Sample Dependence**: The headline conclusion depends on strict benchmark quality and event timing.
- **Interpretation**: CAR drift is present but not statistically decisive in the strongest surviving spec.

## Recommended Next Steps
- **Robustness**: Keep the recovered cached run as the baseline and compare it against alternative match tiers.
- **Drift Analysis**: Separate the short-window drift from the longer-window CAR profile.
- **Documentation**: Keep README and audit summaries synchronized after any future cache-backed rebuild.

## Missing Diagnostics
- `placebo_test_summary.csv`: unavailable in the recovered cache-backed run.
- `robustness_by_subsample.csv`: unavailable in the recovered cache-backed run.
