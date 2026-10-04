# 中国A股盈利意外度量与事件研究诊断框架 | Earnings Surprise Measurement and Event-Study Diagnostics in China A-shares

> **2026-10-04：完成数据扩容，旧版 326 行、CAR +0.30% 摘要仍撤回。** 股票名录由 300 家扩至 5,911 家，严格盈利意外信号由 343 条增至 6,665 条，主检验样本由 328 条增至 6,361 条。扩大覆盖后仍未获得正向公告后漂移证据。[扩容报告](outputs/expanded_20260930/report.md) · [样本漏斗](outputs/expanded_20260930/sample_funnel.csv) · [固定敏感性结果](outputs/expanded_20260930/inference.csv) · [原 300 家审计](outputs/event_validation/report.md)。旧 PDF、演示稿和 `outputs/tables` 保留为历史记录，未同步重编，不应继续引用其结论。

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/LANGUAGE-%E4%B8%AD%E6%96%87-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE 中文"></a>
  <a href="#en"><img src="https://img.shields.io/badge/LANGUAGE-ENGLISH-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="LANGUAGE ENGLISH"></a>
</p>

<a id="zh"></a>

## 简体中文 | [English](#en)

### 当前定位 (Current Positioning)
本项目重建同公司、同报告期、公告日前可得的机构预期，并逐项核对事件时序、交易日历、复权收益、后续盈利公告污染与统计不确定性。事件公告期为 2020-01-01 至 2026-09-30；行情从 2019-01-02 开始，为早期事件提供预热历史。6,665 条严格信号覆盖 2,896 家公司，其中年度报告期 6,648 条、一季报 17 条，半年和三季报仍无严格匹配。未将全年预测代替中报或三季报预测。

### 项目核心功能
- 公告日当天及之后的研报全部排除；每机构最后一条有效预期取中位数，不把全年预期匹配半年累计盈利。
- 首次预告为主事件，修订、快报和正式财报用于后续事件污染诊断；重复行、冲突记录和已知修订不会伪装成独立初始事件。
- 第 0 日固定为公告后首个交易所交易日；缺失或停牌不压缩时间轴，未完整覆盖的窗口保持缺失。
- 复权价格计算个股收益，并保留未复权对照；主检验为清洁事件正、负意外的 CAR[1,10] 均值差，报告公司/月双向聚类置信区间。
- 固定报告 180/365 天新鲜度、1/2 家机构敏感性，不再从 1,920 个规格中挑选“最强”结论。

### 关键结果摘要 (Key Results Snapshot)
| 指标 | 原 300 家审计 | 扩容结果 |
| :--- | ---: | ---: |
| 股票名录 | 300 | 5,911（含 339 家退市公司） |
| 严格信号 / 匹配公司 | 343 / 148 | 6,665 / 2,896 |
| 通过上市时间与流动性筛选 | 333 | 6,487 |
| 无重叠、完整十日收益窗口 | 328 | 6,374 |
| 正负意外主检验事件 | 328 | 6,361 |
| 主检验公司 / 公告月份 | 145 / 18 | 2,834 / 41 |
| 行情截止 | 2026-04-21 | 2026-09-30 |

6,374 个清洁窗口中有 13 个零意外事件，不进入正负分组差检验。主检验正意外减负意外 CAR[1,10] 为 **−1.03%**，95% 公司/月双向聚类区间 **[−2.70%, +0.65%]**，p=0.2231。主检验包含 6,358 个年度事件和 3 个一季报事件，仍以年度预告为主，不支持正向 PEAD，也不能据此断言存在可交易的反向效应。

新增来源包括 42,148 条业绩预告、11,667 条快报、123,894 条利润表记录；匹配公司日线 4,847,159 行、换手率 4,829,592 行、复权因子 4,884,992 行。预告中 41,790 条属于分析采用的人民币股票和财报期。来源总数不等于独立可检验事件数，详见 [扩容口径与断点续跑](DATA_EXPANSION.md)。


### 如何复现 (How to Reproduce)
1. 安装依赖：`pip install -r requirements.txt`。
2. 已有本地冻结来源时：`python main.py` 或 `python scripts/run_full_validation.py`，默认运行扩容样本、离线、不需要 Token。
3. 独立核对全部来源分片、6,665 条事件收益与时序：`python scripts/verify_expanded_data.py`；回归测试：`python -m pytest tests -q`。
4. 获取或继续补数：设置环境变量 `TUSHARE_TOKEN`，依次运行 `python scripts/expand_event_data.py prepare`、`python scripts/expand_event_data.py reports`（额度允许时）、`python scripts/run_event_validation.py --expanded --matching-only`、`python scripts/expand_event_data.py prices`，再运行默认入口。分页和限额规则见 [DATA_EXPANSION.md](DATA_EXPANSION.md)。
5. 原 300 家审计用 `python main.py --baseline` 及 `python scripts/verify_event_validation.py` 复现。授权原始快照只留本地并已 gitignore，遵循 [DATA_LICENSE.md](DATA_LICENSE.md)；远程克隆需自行恢复原始数据，不能宣称无数据即可复现。

旧流程仅能通过 `python main.py --legacy-pipeline` 显式运行。`update_readme_results.py` 检测到当前审计后拒绝用历史表覆盖新结论。

### 数据与研究局限
- 76 个历史月度研报文件中，75 个恰好达到 5,000 行。本轮 `report_rc` 返回每天 10 次额度已耗尽，**新增分析师预测记录为 0**；预测覆盖仍不完整。严格信号增加主要来自扩大股票池和补充公告、行情，不能将其解释为预测源已经补全。
- 名录包括当前可获取的在市、退市股票；严格事件涵盖 34 家退市公司，但历史代码迁移、缺失预测、退市后缺失收益及当时版本仍未完整认证。北交所匹配 85 条、科创板 1,002 条、创业板 1,789 条、主板 3,789 条；覆盖增加仍不代表全市场无偏样本。
- 公告只有日期、无盘前盘后时间戳；个股复权收益与沪深 300 价格指数口径也不完全相同。当前结果是探索性事件研究，不是因果识别或可交易收益。
- 2,896 家匹配公司的 11,584 份行情、换手率、复权、利润表请求全部完成；其中 `603982.SH` 的复权结果为空，两条信号无可用复权收益并被排除。停牌、缺失和未成熟窗口保持缺失，不补零。利润表只用于来源扩充和公告重叠筛查，未混入首次预告主检验。

---

<a id="en"></a>

## English | [中文](#zh)

### Current Positioning
The 2026-10-04 expansion preserves the withdrawal of the old 326-row / +0.30% headline. Strict pre-announcement, same-period signals increase from 343 to 6,665 across 2,896 firms. Events span 2020-01-01 to 2026-09-30, with prices from 2019 for warmup. Matches comprise 6,648 annual and 17 first-quarter preannouncements; semiannual and third-quarter matches remain absent. Use the [expanded audit](outputs/expanded_20260930/report.md); historical PDFs have not been regenerated.

### What This Project Does
- **Expectation Panels**: Builds analyst expectation panels from sell-side reports.
- **Event Construction**: Identifies preannouncements, revisions, express results, and formal releases.
- **Surprise Measurement**: Compares raw, percentage, and standardized surprise metrics.
- **Event-Study Diagnostics**: Computes CARs for leakage, immediate reaction, and post-event drift windows.
- **Robustness Checks**: Evaluates strict vs. relaxed matching and coverage thresholds.

### Key Results Snapshot
| Metric | Value |
| :--- | :--- |
| Initial events / strict matches / clean 10-day windows | 39,586 / 6,665 / 6,374 |
| Primary positive-minus-negative CAR[1,10] | −1.03%; 95% two-way cluster CI [−2.70%, +0.65%]; p=0.2231 |
| Primary coverage | 6,361 nonzero-surprise events; 2,834 firms; 41 announcement months |
| Conclusion | No credible positive PEAD evidence; capped forecast coverage blocks market-wide inference |


### How to Reproduce
Install requirements, restore your licensed frozen inputs, then run `python main.py`, `python scripts/verify_expanded_data.py`, and `python -m pytest tests -q`. The default expanded workflow is offline; `python main.py --baseline` preserves the 300-stock audit. See [source collection and resumability](DATA_EXPANSION.md). Raw snapshots remain local and excluded from Git. The legacy pipeline requires `--legacy-pipeline`; the old README updater cannot overwrite the current audit.

### Limitations
- 75 of 76 monthly forecast caches contain exactly 5,000 rows. No new analyst forecasts were retrieved: the provider reported that its 10-request daily allowance was exhausted. Partial pages are never certified complete.
- The 5,911-name universe includes delisted names, but source gaps, historical identifier changes and unavailable original vintages still limit representativeness.
- All 11,584 bounded requests completed for the 2,896 matched firms. One adjustment-factor snapshot is empty; its two events have missing adjusted returns and are excluded. Financial statements supplement announcement-overlap checks rather than being pooled with preannouncement signals.
- Date-only announcement timing, benchmark conventions and exploratory historical selection remain limitations. No causal efficiency or executable-profit claim is made.

---
Detailed documentation available in `docs/`.
