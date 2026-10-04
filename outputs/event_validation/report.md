# 盈利意外重建审计（2026-10-04）

旧版 326 行、CAR +0.30% 的摘要撤回：它不是冻结同一数据和正确时序后得到的可复核结论。
本轮读取端点缓存，未覆盖旧原始数据/历史结果。价格实际截止 2026-04-21，不是 2026-10-04。

## 已修复的问题

1. 预期截止改为公告日前，公告当天及周末后验研报均排除；同公司、同报告期、每机构最后一条有效预期再取中位数。半年/三季累计业绩不会匹配全年预测。
2. 固定交易所日历，第 0 日为公告后首个交易日；停牌/缺失不压缩窗口，不把不完整收益置零。严格区分 CAR[0,1] 与 CAR[1,10]。
3. 以 close × adj_factor 计算复权日收益；未复权版本只作诊断。保留公告后其他盈利事件污染标记，主检验排除 [-10,+10] 内其他事件。
4. 不再搜索 1,920 个规格后选择“最强”结论；主检验预先写定为清洁事件的正负意外 CAR[1,10] 差。报告双向聚类置信区间及固定敏感性组，其他检验均属探索。
5. 分页抓取报告必须读到空页，完整缓存另存 report_rc_complete；失败月份不被认证完整。修正 tp 误作目标价、快报元/万元混用、扣非净利润混作普通净利润。

## 样本流失

| stage                                      |   count |
|:-------------------------------------------|--------:|
| raw_forecast_rows_300_names                |    4994 |
| conflicting_same_date_rows_excluded        |       9 |
| rows_in_ambiguous_fiscal_periods_excluded  |       9 |
| unique_forecast_rows                       |    3069 |
| initial_forecasts_in_price_span            |    2986 |
| legacy_next_session_cutoff_matched         |     343 |
| legacy_latest_report_on_after_announcement |      67 |
| consensus_changed_after_cutoff_fix         |      61 |
| surprise_sign_changed_after_cutoff_fix     |       4 |
| strict_announcement_cutoff_matched         |     343 |
| prior_liquidity_and_listing_eligible       |     333 |
| matched_events_missing_liquidity_field     |       0 |
| complete_adjusted_CAR_1_10                 |     333 |
| complete_adjusted_CAR_1_10_no_overlap      |     328 |

报告期覆盖：

| period_label   |   events |   matched |   match_rate |
|:---------------|---------:|----------:|-------------:|
| 03-31          |      494 |         0 |     0        |
| 06-30          |     1017 |         0 |     0        |
| 09-30          |      411 |         0 |     0        |
| 12-31          |     1064 |       343 |     0.322368 |

本次所有可匹配事件都是 12 月 31 日全年报告期；其他季度的严格可用预期为零。因此当前证据只能描述年度业绩预告子样本，不能称为覆盖四个季度的盈利意外研究。

## 重算结果

以下均为算术累计市场调整收益、小数单位；例如 0.01 为 1%。CI 为 95% 双向 firm/month 聚类区间。

| window     | statistic               |   n |   firms |   months |   estimate |   ci_low |   ci_high |   p_value |
|:-----------|:------------------------|----:|--------:|---------:|-----------:|---------:|----------:|----------:|
| CAR_-10_-1 | mean_car                | 326 |     145 |       18 |   -0.00694 | -0.03198 |   0.01811 |   0.56670 |
| CAR_-10_-1 | positive_minus_negative | 326 |     145 |       18 |    0.03404 |  0.00787 |   0.06020 |   0.01382 |
| CAR_0_1    | mean_car                | 328 |     145 |       18 |   -0.00013 | -0.01168 |   0.01143 |   0.98194 |
| CAR_0_1    | positive_minus_negative | 328 |     145 |       18 |    0.00798 | -0.00948 |   0.02544 |   0.34820 |
| CAR_1_10   | mean_car                | 328 |     145 |       18 |    0.01615 | -0.00422 |   0.03651 |   0.11269 |
| CAR_1_10   | positive_minus_negative | 328 |     145 |       18 |   -0.01996 | -0.04019 |   0.00028 |   0.05292 |
| CAR_1_20   | mean_car                | 323 |     143 |       17 |    0.03449 |  0.00604 |   0.06294 |   0.02054 |
| CAR_1_20   | positive_minus_negative | 323 |     143 |       17 |   -0.02237 | -0.04461 |  -0.00013 |   0.04880 |
| CAR_1_60   | mean_car                |  37 |      32 |       11 |    0.00043 | -0.08560 |   0.08645 |   0.99136 |
| CAR_1_60   | positive_minus_negative |  37 |      32 |       11 |   -0.02692 | -0.14835 |   0.09450 |   0.63194 |

主检验 N=328，差值=-0.01996，CI=[-0.04019, 0.00028]，p=0.05292。
这里检验的是盈利意外与后续收益的关联，不是总样本平均收益是否为正；平均 CAR 不能证明 PEAD。

同一可用事件的复权前后对照：

| window   |   paired_n |   unadjusted_mean |   adjusted_mean |   events_changed_by_adjustment |
|:---------|-----------:|------------------:|----------------:|-------------------------------:|
| CAR_0_1  |        333 |          -0.00029 |        -0.00029 |                              0 |
| CAR_1_10 |        333 |           0.01698 |         0.01700 |                              1 |
| CAR_1_20 |        332 |           0.03410 |         0.03411 |                              1 |
| CAR_1_60 |        283 |           0.03935 |         0.04092 |                              9 |

## 仍然限制结论的来源缺口

- 历史月度 report_rc 缓存存在截断。本轮是否已按同一 300 股票、同一日期范围逐股完整分页重抓：False。当前使用的疑似截断月份数为 75。本轮修复请求遇到该接口每分钟 1 次的权限限额，已停止报告刷新，未把不完整页面视为完整。完整抓取也不等于“当时可得版本”，供应商可能历史回补；重抓前的独立重算见 `frozen_cache_baseline/`。
- 股票池是当前上市股票代码排序前 300 名，存在幸存者与市场板块偏差；不代表全部 A 股。
- 公告仅有日期，采用次交易日对齐的保守约定；无盘前/盘后时间戳。快报/正式财报缓存可能遗漏后续事件，污染剔除只是已观测事件层面。
- 价格指数基准与含公司行动的个股收益口径不完全对称；双向聚类不能消除这些测量偏差。尚未验证行业/因子基准，不作市场效率或交易获利结论。
- 冻结阈值后的历史回看不等于真正前瞻样本外；180/365 天和 1/2 机构结果全部保留，不选显著项。
- 旧 PDF、演示稿和 outputs/tables 是历史产物，未同步重编；本目录是当前审计版本。

## 复现与文件

`python scripts/run_event_validation.py` 完全离线使用已保存复权因子；首次补因子用 `--refresh-factors`。
`events.csv` 是事件级账本；`matching_audit.csv` 保留失配行与修复前后预期；`sample_funnel.csv`、`period_coverage.csv`、`source_inventory.csv`、`inference.csv` 和 `provenance.json` 可逐项复核。

字段定义：[report_rc](https://tushare.pro/document/2?doc_id=292)、[forecast](https://tushare.pro/document/2?doc_id=45)、[express](https://tushare.pro/document/2?doc_id=46)、[fina_indicator](https://tushare.pro/document/2?doc_id=79)。
