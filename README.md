# 中国A股盈利意外、估值调整与市场有效性研究 | Earnings Surprise, Valuation, and Market Efficiency (China A-share)

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
  <a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文

当前语言：中文 | [Switch to English](#en)

## 项目概述

本仓库已从“弱代理业绩意外 + 单一 guidance 事件”的旧设计，重构为一个以 **Tushare Pro 为主数据源** 的中国 A 股盈利意外研究框架。新版本把卖方预期、管理层业绩预告、业绩快报、正式财务披露和市场控制变量放进同一条可审计的研究流程中。

## 当前推荐主规格

推荐主规格不再以旧的 guidance 代理 surprise 或长窗口 `CAR60` 为核心，而是采用更干净、可审计的 Tushare-first 设计：

- 预期来源：`report_rc`
- 事件来源：`forecast` / `forecast_vip`、`express` / `express_vip`、`fina_indicator`
- 控制变量来源：`daily_basic`
- 主事件窗口：`CAR5`
- 全部输出窗口：`CAR3`、`CAR5`、`CAR10`、`CAR20`、`CAR60`
- 主回归：行业固定效应 + 年季度固定效应 + 事件类型固定效应 + 公司聚类稳健标准误
- 主解释：更严格的预期匹配和事件分类支持对“短期估值调整”的更可信检验，但当前测试样本下的经济幅度与显著性都应谨慎解读

## 为什么旧设计偏弱

旧版本的主信号主要依赖内部代理预期：
- 同公司上一条 guidance
- 公司历史同季度 guidance 均值
- 行业历史中位数
- 零值兜底

这类代理适合做探索，但很难作为主研究设计，因为：
- 与真实卖方一致预期不完全等价；
- 容易引入测量误差；
- 更难严格排除前视偏差；
- 把不同类型事件混在一起解释时更容易过度推断。

因此，新版本把旧 guidance-only 路径保留为 `legacy_guidance` fallback，而把 Tushare-enhanced 路径设为默认主规格。

## Tushare-first 模块映射

### 1. 卖方预期模块
来源：`report_rc`

核心字段：
- `ts_code`
- `report_date`
- `report_title`
- `report_type`
- `classify`
- `org_name`
- `author_name`
- `quarter`
- `np`
- `eps`
- `pe`
- `rating`
- `max_price`
- `min_price`

功能：
- 构建按股票-报告期-发布日期组织的卖方预期面板
- 支持 `latest` / `mean` / `median` 聚合
- 支持最小有效报告数过滤
- 支持从 `report_title` 提取“超预期”辅助标签

### 2. 盈利事件模块
来源：
- `forecast` / `forecast_vip`
- `express` / `express_vip`
- `fina_indicator`

标准化事件类型：
- `preannouncement`
- `revision`
- `express`
- `formal_release`

保留字段：
- `ann_date`
- `end_date`
- `type`
- `p_change_min`
- `p_change_max`
- `net_profit_min`
- `net_profit_max`
- `first_ann_date`
- `summary`
- `change_reason`
- 可用的 YoY 指标

### 3. 市场与控制变量模块
来源：`daily_basic`

核心字段：
- `total_mv`
- `circ_mv`
- `turnover_rate`
- `turnover_rate_f`
- `pe_ttm`
- `pb`
- `ps_ttm`

功能：
- 控制变量
- 流动性筛选
- 微盘/噪声过滤
- 估值控制

## 当前信号体系

新版本按事件类型分别构造 surprise，不再强行把所有事件压成同一种信号：

- `forecast_surprise_raw/pct/std`
- `express_surprise_raw/pct/std`
- `final_surprise_raw/pct/std`
- `revision_magnitude_np`
- `revision_magnitude_eps`
- `upward_revision_count`
- `fraction_upgraded`
- `target_price_change`

主信号为：
- `main_surprise_std`

它优先使用严格 pre-event `report_rc` 匹配得到的 surprise，并在事件家族内部标准化。

## 预期匹配原则

新版本在主样本中执行严格匹配：
- 卖方报告发布日期必须早于事件交易日
- 预期必须匹配同一报告期
- 支持 freshness window
- 支持 `min_valid_report_count`
- 匹配质量会在输出中显式记录，而不是静默回退

## 事件研究与回归设计

### 事件研究
- 异常收益定义：`AR = stock_return - market_return`
- 输出窗口：`CAR3`、`CAR5`、`CAR10`、`CAR20`、`CAR60`

### 推荐回归
对每个 CAR 窗口运行：

`CAR_w ~ main_surprise_std + log_total_mv + beta + book_to_market + turnover20 + pe_ttm + ps_ttm + Industry FE + YearQuarter FE + EventType FE`

标准误：
- 按公司 `ts_code` 聚类

### 审计输出
- 按年份事件数
- 按事件类型事件数
- 按年份信号覆盖率
- 缺失值汇总
- endpoint 能力检测
- Tushare-first update note

## 目录结构

### 原始数据
- `data_raw/`
- `data_raw/tushare/stock_basic/`
- `data_raw/tushare/report_rc/`
- `data_raw/tushare/forecast/`
- `data_raw/tushare/express/`
- `data_raw/tushare/fina_indicator/`
- `data_raw/tushare/daily_basic/`

### 中间结果
- `data_processed/normalized/`
- `data_processed/expectations/`
- `data_processed/events/`
- `data_processed/panels/`

### 最终输出
- `outputs/tables/`
- `outputs/figures/`
- `outputs/audit/`

## 关键代码文件

- `main.py`
- `src/pipeline.py`
- `src/config.py`
- `src/data_collection.py`
- `src/tushare_normalization.py`
- `src/expectation_alignment.py`
- `src/tushare_event_design.py`
- `src/panel_outputs.py`
- `src/guidance_design.py`（legacy fallback）

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 Tushare Token

```bash
export TUSHARE_TOKEN="your_token_here"
```

Windows Git Bash 同样可使用上述写法。不要把 token 写入仓库文件。

### 3. 运行推荐测试规格

```bash
RUN_MODE=test \
FRAMEWORK_MODE=both \
SAMPLE_STOCK_COUNT_TEST=120 \
USE_CACHE=1 \
REQUEST_PAUSE_SEC=0.12 \
python main.py
```

### 4. 只运行新主规格

```bash
RUN_MODE=test \
FRAMEWORK_MODE=tushare_first \
python main.py
```

### 5. 只运行旧 fallback 路径

```bash
RUN_MODE=test \
FRAMEWORK_MODE=legacy_guidance \
python main.py
```

## 主要输出文件

### 新主规格输出
- `outputs/tables/event_dataset_tushare_first.csv`
- `outputs/tables/regression_results_tushare_first.csv`
- `outputs/tables/event_counts_by_type_tushare_first.csv`
- `outputs/tables/event_counts_by_year_tushare_first.csv`
- `outputs/audit/signal_coverage_by_year_tushare_first.csv`
- `outputs/audit/missingness_summary_tushare_first.csv`
- `outputs/audit/audit_note_tushare_first.txt`
- `outputs/audit/tushare_endpoint_capabilities.csv`

### legacy fallback 输出
- `outputs/tables/final_dataset_legacy_guidance.csv`
- `outputs/tables/final_regression_results_legacy_guidance.csv`
- `outputs/tables/final_interpretation_legacy_guidance.txt`

### 汇总输出
- `outputs/tables/run_summary.csv`
- `outputs/audit/tushare_first_update_note.txt`

## 当前结论口径

当前更合适的 headline 不是：
- 强 PEAD
- 长期漂移是主证据
- 所有事件类型都应该合并解释

更合适的表述是：

> 在更严格的 Tushare-first 研究设计下，盈利相关事件相对于卖方预期的 surprise 提供了更干净的短期估值调整检验框架，但当前测试样本下的经济幅度、显著性与覆盖率都应谨慎解读。

## 备注

- `forecast_vip`、`express_vip` 和 `report_rc` 的可用性取决于当前 Tushare 账户权限，仓库会把 endpoint 可用性写入审计输出。
- 新版本优先使用严格的 `report_rc` 匹配样本；弱匹配或 fallback 样本应作为 supplementary 结果解读。
- 旧 guidance-only 设计保留是为了可重复旧结果，不应继续作为默认 headline。

---

<a id="en"></a>

## English

Current language: English | [切换到中文](#zh)

## Project Overview

This repository has been upgraded from a weak-proxy guidance-only event study into a **Tushare Pro-first earnings-surprise research framework** for China A-shares. The revised design integrates sell-side expectations, management preannouncements, earnings express releases, formal financial releases, and market controls into one auditable pipeline.

## Current Preferred Specification

The headline specification no longer relies on the old guidance-only proxy or long-window `CAR60`. The preferred design is now:

- expectation source: `report_rc`
- event sources: `forecast` / `forecast_vip`, `express` / `express_vip`, `fina_indicator`
- market/control source: `daily_basic`
- headline event window: `CAR5`
- full event-window grid: `CAR3`, `CAR5`, `CAR10`, `CAR20`, `CAR60`
- main regression: industry FE + year-quarter FE + event-type FE + firm-clustered SE
- interpretation: cleaner expectation alignment and event classification support a more credible short-run valuation-adjustment test, but the current test-run evidence remains cautious rather than strong

## Why the older design was weak

The older design built expected earnings from internal proxies such as:
- prior guidance for the same firm-period,
- firm historical guidance,
- industry medians,
- or zero fallback.

That can be useful for exploration, but it is not a strong main design because it does not map cleanly into true pre-event sell-side expectations, introduces measurement error, and makes look-ahead control harder. The old guidance-only path is therefore retained as a `legacy_guidance` fallback rather than the headline result.

## Tushare-first module mapping

### 1. Sell-side expectation module
Source: `report_rc`

Core fields include:
- `ts_code`
- `report_date`
- `report_title`
- `report_type`
- `classify`
- `org_name`
- `author_name`
- `quarter`
- `np`
- `eps`
- `pe`
- `rating`
- `max_price`
- `min_price`

Functions:
- build a stock-period-date sell-side expectation panel
- support `latest` / `mean` / `median` aggregation
- enforce a minimum report count filter
- extract supplementary over-expectation labels from report titles

### 2. Earnings-event module
Sources:
- `forecast` / `forecast_vip`
- `express` / `express_vip`
- `fina_indicator`

Standardized event types:
- `preannouncement`
- `revision`
- `express`
- `formal_release`

### 3. Market/control module
Source: `daily_basic`

Core fields:
- `total_mv`
- `circ_mv`
- `turnover_rate`
- `turnover_rate_f`
- `pe_ttm`
- `pb`
- `ps_ttm`

## Current signal system

The revised framework builds surprise measures by event family rather than forcing all events into one noisy pool:

- `forecast_surprise_raw/pct/std`
- `express_surprise_raw/pct/std`
- `final_surprise_raw/pct/std`
- `revision_magnitude_np`
- `revision_magnitude_eps`
- `upward_revision_count`
- `fraction_upgraded`
- `target_price_change`

The main signal is:
- `main_surprise_std`

It prioritizes strictly pre-event `report_rc` matches and standardizes within comparable event families.

## Matching rules

The preferred sample enforces:
- sell-side reports must be published before the event trading date,
- matched expectations must refer to the same fiscal period,
- freshness windows are configurable,
- minimum report-count filters are configurable,
- match quality is recorded explicitly rather than hidden in silent fallback logic.

## Event-study and regression design

### Event study
- abnormal return: `AR = stock_return - market_return`
- windows: `CAR3`, `CAR5`, `CAR10`, `CAR20`, `CAR60`

### Preferred regression
For each CAR window, the preferred model is:

`CAR_w ~ main_surprise_std + log_total_mv + beta + book_to_market + turnover20 + pe_ttm + ps_ttm + Industry FE + YearQuarter FE + EventType FE`

Standard errors:
- clustered at the firm (`ts_code`) level

### Audit outputs
The revised pipeline produces:
- event counts by year
- event counts by type
- signal coverage by year
- missingness summaries
- endpoint capability checks
- a Tushare-first update note

## Key code files

- `main.py`
- `src/pipeline.py`
- `src/config.py`
- `src/data_collection.py`
- `src/tushare_normalization.py`
- `src/expectation_alignment.py`
- `src/tushare_event_design.py`
- `src/panel_outputs.py`
- `src/guidance_design.py` (legacy fallback)

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set the Tushare token

```bash
export TUSHARE_TOKEN="your_token_here"
```

Do not store the token in repo files.

### 3. Run the recommended test specification

```bash
RUN_MODE=test \
FRAMEWORK_MODE=both \
SAMPLE_STOCK_COUNT_TEST=120 \
USE_CACHE=1 \
REQUEST_PAUSE_SEC=0.12 \
python main.py
```

### 4. Run only the new preferred framework

```bash
RUN_MODE=test \
FRAMEWORK_MODE=tushare_first \
python main.py
```

### 5. Run only the legacy fallback path

```bash
RUN_MODE=test \
FRAMEWORK_MODE=legacy_guidance \
python main.py
```

## Main outputs

### Preferred Tushare-first outputs
- `outputs/tables/event_dataset_tushare_first.csv`
- `outputs/tables/regression_results_tushare_first.csv`
- `outputs/tables/event_counts_by_type_tushare_first.csv`
- `outputs/tables/event_counts_by_year_tushare_first.csv`
- `outputs/audit/signal_coverage_by_year_tushare_first.csv`
- `outputs/audit/missingness_summary_tushare_first.csv`
- `outputs/audit/audit_note_tushare_first.txt`
- `outputs/audit/tushare_endpoint_capabilities.csv`

### Legacy fallback outputs
- `outputs/tables/final_dataset_legacy_guidance.csv`
- `outputs/tables/final_regression_results_legacy_guidance.csv`
- `outputs/tables/final_interpretation_legacy_guidance.txt`

### Run summary outputs
- `outputs/tables/run_summary.csv`
- `outputs/audit/tushare_first_update_note.txt`

## Recommended final framing

A better headline is:

> Under a cleaner Tushare-first design, earnings-related surprises relative to sell-side expectations can be studied with a more credible short-run valuation-adjustment framework in China A-shares, but the current test-run magnitude, significance, and coverage should still be interpreted cautiously.

## Notes

- Availability of `forecast_vip`, `express_vip`, and `report_rc` depends on the current Tushare account entitlements; the pipeline records endpoint availability in the audit outputs.
- The strict `report_rc`-matched sample should be treated as the main result; weaker matches or fallback samples should be supplementary.
- The old guidance-only design remains only for reproducibility and comparison, not as the default headline result.
