# 中国A股盈利意外度量与事件研究诊断框架 | Earnings Surprise Measurement and Event-Study Diagnostics in China A-shares

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
  <a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文 | [English](#en)

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
   - 复制模板: `cp .env.example .env` 或 PowerShell `Copy-Item .env.example .env`
   - 在 `.env` 中填写您的 `TUSHARE_TOKEN`，脚本会自动加载仓库根目录的 `.env`
   - **安全提醒**: 切勿提交 `.env` 或 `.claude/settings.local.json` 到版本控制。使用 `.env.example` 作为模板。如果 Token 意外泄露，请立即在 Tushare Pro 后台重置。
3. **运行验证流程**:
   - 冒烟测试: `python scripts/run_smoke_test.py`
   - 全量实证流程: `python scripts/run_full_validation.py`
   - 更新 README 结果: `python scripts/update_readme_results.py`
4. **运行测试**: `pytest`

### 数据缓存与刷新
- **缓存机制**: 数据采集器按股票代码和日期范围缓存 Tushare 响应至 `data_raw/cache/`。缓存键已包含 `start_date` 和 `end_date`，修改日期范围后会自动获取新区间数据。
- **强制刷新**: 设置 `FORCE_REFRESH=1` 可忽略所有缓存，重新获取全部数据。适用于 Token 权限变更或怀疑数据陈旧时。
- **数据新鲜度审计**: 每次完整运行后，`outputs/audit/data_freshness_audit.csv` 记录各端点的行数、最大交易日期及运行时间戳。

### 已知局限
- **Tushare 权限**: 数据的深度和广度受账户积分限制。`fina_indicator` 端点每日限额约 500 次调用，全量 1200 只股票需分批获取。
- **预期偏差**: 分析师预期可能存在滞后或乐观偏差（中位数滞后约 61 天）。
- **公告效应**: 盘后公告的即时反应可能映射到次日，导致事件窗口定义存在模糊性。
- **样本量**: 严格匹配（strict same quarter）的可用样本仅约 326 条，限制了统计推断的效力。
- **稳健性**: Placebo 检验和子样本稳健性分析已实现基础版本，结果保存至 `outputs/tables/`。数据不足时自动降级为说明文件。
- **结论解读**: 实证结果不应被过度解读为市场效率的定论。本项目定位为诊断框架而非最终结论。

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
   - Copy template: `cp .env.example .env` or PowerShell `Copy-Item .env.example .env`
   - Fill in your `TUSHARE_TOKEN` in `.env`; the scripts automatically load the repository-root `.env`
   - **Security note**: Never commit `.env` or `.claude/settings.local.json` to version control. Use `.env.example` as the template. If a token is accidentally exposed, rotate it immediately in Tushare Pro.
3. **Validation Workflow**:
   - Smoke Test: `python scripts/run_smoke_test.py`
   - Full Empirical Pipeline: `python scripts/run_full_validation.py`
   - Update README Results: `python scripts/update_readme_results.py`
4. **Run Tests**: `pytest`

### Data Caching and Freshness
- **Cache mechanism**: The data collector caches Tushare responses per stock and date range under `data_raw/cache/`. Cache keys now include `start_date` and `end_date`, so changing the date range automatically fetches new data for the extended window.
- **Force refresh**: Set `FORCE_REFRESH=1` to bypass all caches and re-fetch everything. Useful when token permissions change or data staleness is suspected.
- **Freshness audit**: Each full run writes `outputs/audit/data_freshness_audit.csv` recording row counts, max trade dates, and run timestamp for every endpoint.

### Known Limitations
- **Tushare Permissions**: Coverage depth depends on Tushare Pro account tier. The `fina_indicator` endpoint has a daily limit of ~500 calls; full 1200-stock refresh requires batched runs.
- **Expectation Quality**: Analyst forecasts may be stale (median lag ~61 days) or optimistically biased.
- **Event Timing**: After-market announcements may cause reaction lags, introducing ambiguity in event-window alignment.
- **Sample Size**: Strict-match (same quarter) yields only ~326 usable signal rows, limiting statistical power.
- **Robustness Diagnostics**: Basic placebo tests and subsample robustness splits are generated and saved to `outputs/tables/`. When data is insufficient, note files explain unavailability instead.
- **Interpretation**: Empirical results are diagnostic evidence, not a final proof of A-share market efficiency or inefficiency. This project is a measurement and event-study auditing framework.

---
Detailed documentation available in `docs/`.
