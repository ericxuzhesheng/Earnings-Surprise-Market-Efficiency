# 中国A股盈利意外度量与事件研究诊断框架 | Earnings Surprise Measurement and Event-Study Diagnostics in China A-shares

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
  <a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文 | [English](#en)

### 当前定位 (Current Positioning)
本项目是一个基于 Tushare Pro 数据源的**诊断框架**，旨在展示盈利意外度量、事件定义、预期匹配质量以及覆盖率筛选如何共同影响中国 A 股市场的实证金融推断。本项目不应被视为市场无效性的最终证据，而应作为研究盈利事件时的基准审计工具。

### 项目核心功能
- **预期面板构建**：整合卖方分析师预期（`report_rc`）。
- **盈利事件识别**：覆盖业绩预告、修订、快报及正式财报。
- **多维意外度量**：并行比较 Raw、Percentage 与 Standardized Surprise。
- **事件研究诊断**：计算泄露窗（Leakage）、即时反应窗与漂移窗（Drift）的累计异常收益（CAR）。
- **稳健性测试**：提供不同匹配层级（Strict vs Relaxed）的对比分析。

### 关键结果摘要 (Key Results Snapshot - PENDING)
> **注意**：当前环境未配置 `TUSHARE_TOKEN`，以下表格仅为占位符。全量验证后的真实数据将在此展示。

| 指标 | 结果 |
| :--- | :--- |
| 基准样本量 (Headline Sample Size) | PENDING |
| 主事件类型 (Main Event Type) | 业绩预告 (Preannouncement) |
| 匹配层级 (Matching Tier) | 严格同季度匹配 (Strict Same Quarter) |
| CAR[1, 10] 高低组价差 | PENDING |
| 泄露/漂移证据 | PENDING |

### 如何复现 (How to Reproduce)
1. **安装依赖**: `pip install -r requirements.txt`
2. **设置 Token**: `export TUSHARE_TOKEN="your_token_here"`
3. **运行冒烟测试**: `python scripts/run_smoke_test.py`
4. **运行全量流程**: `python main.py`

### 数据与研究局限
- **Tushare 权限**: 数据的深度和广度受账户积分限制。
- **预期偏差**: 分析师预期可能存在滞后或乐观偏差。
- **公告效应**: 盘后公告的即时反应可能映射到次日。
- **结论解读**: 实证结果不应被过度解读为市场效率的定论。

---

<a id="en"></a>

## English | [中文](#zh)

### Current Positioning
This project provides a **diagnostic framework** for China A-share earnings surprise research using Tushare Pro. It demonstrates how measurement choices (surprise scaling, event definition, match quality) affect empirical inference. It is a baseline tool for auditing event-study designs, not a final proof of market inefficiency.

### What This Project Does
- **Expectation Panels**: Builds analyst expectation panels from sell-side reports.
- **Event Construction**: Identifies preannouncements, revisions, express results, and formal releases.
- **Surprise Measurement**: Compares raw, percentage, and standardized surprise metrics.
- **Event-Study Diagnostics**: Computes CARs for leakage, immediate reaction, and post-event drift windows.
- **Robustness Checks**: Evaluates strict vs. relaxed matching and coverage thresholds.

### Key Results Snapshot - PENDING
> **Note**: `TUSHARE_TOKEN` is missing in the current environment. The table below is a placeholder. Real results from a full validation run will be populated here.

| Metric | Value |
| :--- | :--- |
| Final Headline Sample Size | PENDING |
| Main Event Type | Preannouncement |
| Matching Tier | Strict Same Quarter |
| CAR[1, 10] High-Minus-Low Spread | PENDING |
| Leakage/Drift Evidence | PENDING |

### How to Reproduce
1. **Dependencies**: `pip install -r requirements.txt`
2. **Token**: `export TUSHARE_TOKEN="your_token_here"`
3. **Smoke Test**: `python scripts/run_smoke_test.py`
4. **Full Pipeline**: `python main.py`

### Limitations
- **Data Access**: Coverage depends on Tushare Pro permission tiers.
- **Expectation Quality**: Analyst forecasts may be stale or biased.
- **Timing**: Announcements after market close may cause reaction lags.
- **Interpretation**: Evidence should not be overinterpreted as a final market-efficiency conclusion.

---
Detailed documentation available in `docs/`.
