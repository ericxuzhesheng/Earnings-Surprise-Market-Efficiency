# 中国A股盈利意外、估值调整与市场有效性研究 | Earnings Surprise, Valuation, and Market Efficiency (China A-share)

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
  <a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文

当前语言：中文 | [Switch to English](#en)

## 项目目标

本项目研究中国A股（2020年至今）中，盈利相关业绩预告事件是否对应公告后异常收益，并从公司金融视角解释其估值调整与市场有效性含义。

## 最新主结论

本项目最新推荐的主规格不再以长窗口 CAR60 或“强PEAD”作为核心叙事，而是聚焦更干净、学术上更可辩护的短窗口估值调整检验：

- 主信号：`ES_std`
- 主结果窗口：`CAR5`
- 主回归：行业固定效应 + 年-季度固定效应 + 公司聚类稳健标准误
- 主解释：更干净的标准化业绩预告意外，能够预测幅度 modest 但统计显著的短期异常收益
- 最终叙事：**cleaner standardized guidance surprise predicts limited short-run valuation adjustment**

这意味着：
- 我们仍然观察到信息进入价格的延迟调整；
- 但证据更支持“有限的短期估值调整”，而不是“强烈的长期漂移”或“显著的市场无效”。

## 方法更新说明

### 旧基准结果（baseline）

旧版本的主规格为：
- 信号：`ES_main = guidance_yoy_midpoint - analyst_consensus_yoy_proxy`
- 结果窗口：`CAR60`
- 回归：基于虚拟变量的 pooled OLS / clustered OLS

该版本的问题是：
- `ES_main` 对预期的代理噪声较大；
- `CAR60` 容易混入大量与公告无关的后续新闻；
- 长窗口结果更容易被误解为强PEAD或强市场低效。

### 新推荐规格（preferred）

新版本将主规格更新为：
- 信号：`ES_std`
- 事件窗口：`CAR5`
- 回归：`CAR5 ~ ES_std + log_size + beta + Industry FE + YearQuarter FE`，并按公司聚类标准误

这样做的原因：
- `ES_std` 相比 `ES_main` 更能控制 surprise 尺度差异；
- `CAR5` 比 `CAR60` 更接近公告信息的直接价格反应；
- 固定效应和聚类标准误能更好地控制行业/时间共同冲击与公司层面的相关性。

## 当前研究设计摘要

- 样本区间：2020-01-01 至 2026-04-21
- 事件类型：
  - `guidance_initial`
  - `guidance_upward_revision`
- 样本过滤：
  - 剔除 ST / *ST
  - 上市交易日少于 120 天剔除
  - 事件日无交易或无成交量剔除
  - 使用 20 日换手率流动性筛选
- 异常收益定义：
  - `AR = stock_return - market_return`
- 当前输出窗口：
  - baseline: `CAR20`, `CAR60`
  - improved: `CAR3`, `CAR5`, `CAR20`

## 核心结果：旧基准 vs 新主规格

### 本次成功运行样本（test mode）

- 股票数：300
- 原始 guidance 行数：5,162
- 事件数（全部候选）：2,858
- baseline 事件数据：2,375
- improved 事件数据：2,529

见：`outputs/tables/run_summary.csv`

### 旧基准结果（baseline）

主结果：`CAR60` 上 moderate positive dummy 为正且边际显著，但该设计不再作为主叙事。

- baseline sample size: 2,375
- `moderate_positive_ES_dummy` on `CAR60`
  - coef = `0.022523`
  - p-value = `0.0481`
- `outputs/tables/final_regression_results_baseline.csv`

组别均值：
- all events `CAR60`: 3.288%
- positive ES `CAR60`: 4.581%
- moderate positive `CAR60`: 4.895%
- extreme ES `CAR60`: 6.981%

见：`outputs/tables/final_group_summary_baseline.csv`

### 新推荐规格结果（improved）

主结果：`ES_std` 对 `CAR5` 的系数为正且统计显著，但经济幅度 modest。

- improved sample size: 2,529
- regression: industry FE + year-quarter FE + firm-clustered SE
- `ES_std` on `CAR5`
  - coef = `0.001732`
  - p-value = `0.0195`
  - `R^2 = 0.272`
- `outputs/tables/final_regression_results_improved.csv`

组别均值：
- all events `CAR5`: -0.176%
- positive ES `CAR5`: -0.127%
- moderate positive `CAR5`: -0.180%
- extreme ES `CAR5`: 0.158%

见：`outputs/tables/final_group_summary_improved.csv`

## 如何解释最新结果

当前最合理的解释不是：
- “强长期漂移已经被清楚识别”，也不是
- “中等正向 surprise 一定优于极端 surprise”，更不是
- “市场明显无效”。

更合理的解释是：
- 在更干净的信号定义和更短的事件窗口下，标准化后的 guidance surprise 对短期异常收益具有统计意义；
- 但该效应经济上并不大；
- 因此最终结论应聚焦于**有限的短期估值调整**，而不是 dramatic market inefficiency。

## 推荐展示口径

### PPT / 口头汇报主结论

> Cleaner standardized guidance surprise predicts modest but statistically significant short-run abnormal return in China A-shares.

中文可表述为：

> 在更可辩护的研究设计下，标准化后的业绩预告意外能够预测幅度有限但统计显著的短期异常收益，说明市场对该类信息存在有限的短期延迟调整。

### 不建议继续作为 headline 的表述

- 强 PEAD
- 长期漂移是主证据
- CAR60 是核心结果
- moderate positive 一定优于 extreme
- 强市场无效

这些说法现在只适合放在 baseline / supplementary / cautionary comparison 中。

## 主要输出文件

### 推荐主结果文件

- `outputs/tables/final_dataset_improved.csv`
- `outputs/tables/final_group_summary_improved.csv`
- `outputs/tables/final_regression_results_improved.csv`
- `outputs/tables/final_interpretation_improved.txt`

### 对比与审计文件

- `outputs/tables/final_dataset_baseline.csv`
- `outputs/tables/final_group_summary_baseline.csv`
- `outputs/tables/final_regression_results_baseline.csv`
- `outputs/tables/final_interpretation_baseline.txt`
- `outputs/tables/before_after_method_comparison.csv`
- `outputs/tables/before_after_interpretation.txt`
- `outputs/tables/project_narrative_update_note.txt`

### 图形文件

- `outputs/figures/fig1_es_group_comparison_improved.png`
- `outputs/figures/fig2_cum_return_moderate_vs_extreme_improved.png`
- `outputs/figures/fig3_event_type_comparison_improved.png`
- `outputs/figures/fig1_es_group_comparison_baseline.png`
- `outputs/figures/fig2_cum_return_moderate_vs_extreme_baseline.png`
- `outputs/figures/fig3_event_type_comparison_baseline.png`

## 运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行当前推荐测试规格：

```bash
RUN_MODE=test
SAMPLE_STOCK_COUNT_TEST=300
LIQUIDITY_TURNOVER20_NEW=0.3
USE_CACHE=1
python main.py
```

## 备注

- 当前 test mode 默认样本为 300 只股票，以减少 Tushare forecast 接口限频问题。
- 若后续获得更高质量分析师一致预期，可进一步替换或增强 `ES_std` 的 expectation benchmark。
- baseline 结果仍保留，用于方法对比，不建议继续作为主结果展示。

## 来源说明

- 数据来源：项目本地构建的中国A股业绩预告事件样本与对应行情数据（2020-01-01 至 2026-04-21）。
- 图表来源：由本项目代码运行后在 `outputs/figures` 与 `outputs/tables` 中自动生成。
- 参考文献：20200930-国信证券-金融工程专题研究：超预期投资全攻略。

---

<a id="en"></a>

## English

Current language: English | [切换到中文](#zh)

## Project Objective

This project studies whether earnings-guidance-related events in China A-shares (2020 onward) are associated with post-announcement abnormal returns, and interprets the evidence from a Corporate Finance perspective centered on valuation adjustment and market efficiency.

## Updated Headline Finding

The project no longer treats long-window `CAR60` or strong PEAD language as the default narrative. The preferred specification now focuses on a cleaner and more defensible short-window design:

- Main signal: `ES_std`
- Main outcome window: `CAR5`
- Main regression: industry fixed effects + year-quarter fixed effects + firm-clustered standard errors
- Main interpretation: cleaner standardized guidance surprise predicts modest but statistically significant short-run abnormal return
- Final framing: **limited short-run valuation adjustment after cleaner guidance surprise**

This means the evidence supports delayed adjustment in prices, but not a strong claim of long-run drift or dramatic market inefficiency.

## Method Update

### Old baseline result

The old baseline used:
- signal: `ES_main = guidance_yoy_midpoint - analyst_consensus_yoy_proxy`
- outcome: `CAR60`
- regression: pooled / clustered OLS using dummy-style signals

Why it is no longer preferred:
- `ES_main` is noisy because expected earnings are measured with a weak proxy;
- `CAR60` absorbs a great deal of non-event news;
- long-window results are easier to over-interpret as strong PEAD.

### New preferred specification

The current recommended specification is:
- signal: `ES_std`
- outcome: `CAR5`
- regression: `CAR5 ~ ES_std + log_size + beta + Industry FE + YearQuarter FE`, with firm-clustered SE

Why it is more defensible:
- `ES_std` better standardizes surprise intensity;
- `CAR5` is closer to the direct valuation response to the guidance event;
- fixed effects and clustered inference better control for common shocks and within-firm dependence.

## Current Research Design Summary

- Sample period: 2020-01-01 to 2026-04-21
- Event types:
  - `guidance_initial`
  - `guidance_upward_revision`
- Filters:
  - exclude ST / *ST firms
  - exclude firms with fewer than 120 trading days since listing
  - exclude non-tradable event-day observations
  - apply liquidity screen based on 20-day turnover
- Abnormal return:
  - `AR = stock_return - market_return`
- Event windows currently produced:
  - baseline: `CAR20`, `CAR60`
  - improved: `CAR3`, `CAR5`, `CAR20`

## Main Results: Baseline vs Preferred Specification

### Successful current run (test mode)

- sample stocks: 300
- raw guidance rows: 5,162
- total candidate events: 2,858
- baseline event dataset: 2,375
- improved event dataset: 2,529

See: `outputs/tables/run_summary.csv`

### Old baseline result

The baseline still shows a positive borderline-significant coefficient on `CAR60`, but this is no longer the headline result.

- baseline sample size: 2,375
- `moderate_positive_ES_dummy` on `CAR60`
  - coef = `0.022523`
  - p-value = `0.0481`
- file: `outputs/tables/final_regression_results_baseline.csv`

Group means:
- all events `CAR60`: 3.288%
- positive ES `CAR60`: 4.581%
- moderate positive `CAR60`: 4.895%
- extreme ES `CAR60`: 6.981%

See: `outputs/tables/final_group_summary_baseline.csv`

### New preferred result

The preferred result is the short-window FE regression using standardized surprise.

- improved sample size: 2,529
- regression: industry FE + year-quarter FE + firm-clustered SE
- `ES_std` on `CAR5`
  - coef = `0.001732`
  - p-value = `0.0195`
  - `R^2 = 0.272`
- file: `outputs/tables/final_regression_results_improved.csv`

Group means:
- all events `CAR5`: -0.176%
- positive ES `CAR5`: -0.127%
- moderate positive `CAR5`: -0.180%
- extreme ES `CAR5`: 0.158%

See: `outputs/tables/final_group_summary_improved.csv`

## Interpretation

The most defensible conclusion is not:
- “strong PEAD is clearly established,”
- or “long-run drift is the main evidence,”
- or “moderate positive surprise clearly dominates extreme surprise,”
- or “the market is strongly inefficient.”

Instead, the evidence supports a narrower conclusion:
- with cleaner signal construction and a shorter event window, standardized guidance surprise helps explain short-run abnormal return;
- the effect is statistically meaningful but economically modest;
- therefore the project should be framed as evidence of **limited short-run valuation adjustment**, not dramatic inefficiency.

## Recommended Presentation Language

### Headline conclusion

> Cleaner standardized guidance surprise predicts modest but statistically significant short-run abnormal return in China A-shares.

### Phrases that should no longer be used as the main takeaway

- strong PEAD
- long-run drift as the main evidence
- CAR60 as the headline result
- moderate positive beats extreme
- strong market inefficiency

These can remain only in baseline or supplementary discussion.

## Main Output Files

### Preferred result files

- `outputs/tables/final_dataset_improved.csv`
- `outputs/tables/final_group_summary_improved.csv`
- `outputs/tables/final_regression_results_improved.csv`
- `outputs/tables/final_interpretation_improved.txt`

### Comparison and audit files

- `outputs/tables/final_dataset_baseline.csv`
- `outputs/tables/final_group_summary_baseline.csv`
- `outputs/tables/final_regression_results_baseline.csv`
- `outputs/tables/final_interpretation_baseline.txt`
- `outputs/tables/before_after_method_comparison.csv`
- `outputs/tables/before_after_interpretation.txt`
- `outputs/tables/project_narrative_update_note.txt`

### Figures

- `outputs/figures/fig1_es_group_comparison_improved.png`
- `outputs/figures/fig2_cum_return_moderate_vs_extreme_improved.png`
- `outputs/figures/fig3_event_type_comparison_improved.png`
- `outputs/figures/fig1_es_group_comparison_baseline.png`
- `outputs/figures/fig2_cum_return_moderate_vs_extreme_baseline.png`
- `outputs/figures/fig3_event_type_comparison_baseline.png`

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the current preferred test configuration:

```bash
RUN_MODE=test
SAMPLE_STOCK_COUNT_TEST=300
LIQUIDITY_TURNOVER20_NEW=0.3
USE_CACHE=1
python main.py
```

## Notes

- Test mode now defaults to 300 stocks to reduce Tushare forecast rate-limit failures.
- If better analyst-consensus data becomes available, the expectation benchmark behind `ES_std` should be upgraded further.
- Baseline results are preserved for methodological comparison, but should not remain the main storyline.

## Source Note

- Data source: locally constructed China A-share guidance-event sample and matched market data (2020-01-01 to 2026-04-21).
- Figures/tables: generated automatically by this project in `outputs/figures` and `outputs/tables`.
- Reference: Guosen Securities (2020-09-30), Financial Engineering Special Research: A Complete Guide to Surprise Investing.
