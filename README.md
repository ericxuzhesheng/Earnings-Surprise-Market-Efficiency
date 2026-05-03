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

### 关键结果摘要 (Key Results Snapshot)
| 指标 | 结果 |
| :--- | :--- |
| 基准样本量 (Headline Sample Size) | 326 (可用信号行) |
| 主事件类型 (Main Event Type) | 业绩预告 (Preannouncement) |
| 匹配层级 (Matching Tier) | Strict Same Quarter |
| CAR[1, 10] 均值 (N=326) | +0.30% (诊断性证据) |
| 泄露/漂移结论 | 样本量过小，结论不具有统计显著性 (详见 RESULTS_AUDIT.md) |


### 如何复现 (How to Reproduce)
1. **安装依赖**: `pip install -r requirements.txt`
2. **设置 Token**: 
   - 复制模板: `cp .env.example .env`
   - 在 `.env` 中填写您的 `TUSHARE_TOKEN`
3. **运行验证流程**:
   - 冒烟测试: `python scripts/run_smoke_test.py`
   - 全量实证流程: `python scripts/run_full_validation.py`
   - 更新 README 结果: `python scripts/update_readme_results.py`
4. **运行测试**: `pytest`

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

### Key Results Snapshot
| Metric | Value |
| :--- | :--- |
| Final Headline Sample Size | 326 (usable signal rows) |
| Main Event Type | Preannouncement |
| Matching Tier | Strict Same Quarter |
| CAR[1, 10] Mean (N=326) | +0.30% (diagnostic evidence) |
| Leakage/Drift Conclusion | Inconclusive due to small N (See RESULTS_AUDIT.md) |


### How to Reproduce
1. **Dependencies**: `pip install -r requirements.txt`
2. **Token Setup**:
   - Copy template: `cp .env.example .env`
   - Fill in your `TUSHARE_TOKEN` in `.env`
3. **Validation Workflow**:
   - Smoke Test: `python scripts/run_smoke_test.py`
   - Full Empirical Pipeline: `python scripts/run_full_validation.py`
   - Update README Results: `python scripts/update_readme_results.py`
4. **Run Tests**: `pytest`

### Limitations
- **Data Access**: Coverage depends on Tushare Pro permission tiers.
- **Expectation Quality**: Analyst forecasts may be stale or biased.
- **Timing**: Announcements after market close may cause reaction lags.
- **Interpretation**: Evidence should not be overinterpreted as a final market-efficiency conclusion.

---
Detailed documentation available in `docs/`.
