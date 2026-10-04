# 盈利意外数据扩容（2026-10-04）

股票池由原300家扩至 5,911 家可获取的人民币股票，包含在市、退市和暂停上市记录（状态数：{'D': 339, 'L': 5572}）。价格区间 2019-01-02 至 2026-09-30。
旧300家冻结审计保留在 `outputs/event_validation/`。本目录仅增加来源覆盖，保持公告前同报告期匹配、365天新鲜度、1家机构、0.3%换手率及120交易日上市年龄的既定规则。

## 样本增加与流失

| stage                                      |   count |
|:-------------------------------------------|--------:|
| raw_forecast_rows_all_available_A_shares   |   41790 |
| conflicting_same_date_rows_excluded        |       0 |
| rows_in_ambiguous_fiscal_periods_excluded  |       0 |
| unique_forecast_rows                       |   41790 |
| initial_forecasts_in_event_span            |   39586 |
| legacy_next_session_cutoff_matched         |    6790 |
| legacy_latest_report_on_after_announcement |    1561 |
| consensus_changed_after_cutoff_fix         |    1305 |
| surprise_sign_changed_after_cutoff_fix     |      91 |
| strict_announcement_cutoff_matched         |    6665 |
| prior_liquidity_and_listing_eligible       |    6487 |
| matched_events_missing_liquidity_field     |       0 |
| complete_adjusted_CAR_1_10                 |    6472 |
| complete_adjusted_CAR_1_10_no_overlap      |    6374 |

| period_label   |   events |   matched |   match_rate |
|:---------------|---------:|----------:|-------------:|
| 03-31          |     4904 |        17 |   0.00346656 |
| 06-30          |    12501 |         0 |   0          |
| 09-30          |     3929 |         0 |   0          |
| 12-31          |    18252 |      6648 |   0.364234   |

## 固定检验

以下为小数单位。主检验正负意外 CAR[1,10] 差 N=6361，估计 -0.01026，95% CI [-0.02702, 0.00650]，p=0.22310。清洁窗口计数包含零意外，正负组主检验排除零意外。平均CAR不等于PEAD，增加样本不保证结果更好。

| window     | statistic               |    n |   firms |   months |   estimate |   ci_low |   ci_high |   p_value |
|:-----------|:------------------------|-----:|--------:|---------:|-----------:|---------:|----------:|----------:|
| CAR_-10_-1 | mean_car                | 6370 |    2836 |       41 |   -0.00989 | -0.04056 |   0.02078 |   0.51823 |
| CAR_-10_-1 | positive_minus_negative | 6357 |    2834 |       41 |    0.04023 |  0.01800 |   0.06245 |   0.00073 |
| CAR_0_1    | mean_car                | 6379 |    2836 |       41 |   -0.00329 | -0.01294 |   0.00635 |   0.49410 |
| CAR_0_1    | positive_minus_negative | 6366 |    2834 |       41 |    0.01378 |  0.00505 |   0.02252 |   0.00278 |
| CAR_1_10   | mean_car                | 6374 |    2836 |       41 |    0.00812 | -0.01737 |   0.03362 |   0.52327 |
| CAR_1_10   | positive_minus_negative | 6361 |    2834 |       41 |   -0.01026 | -0.02702 |   0.00650 |   0.22310 |
| CAR_1_20   | mean_car                | 5373 |    2523 |       36 |    0.02690 |  0.00325 |   0.05054 |   0.02694 |
| CAR_1_20   | positive_minus_negative | 5361 |    2522 |       36 |   -0.00882 | -0.02798 |   0.01035 |   0.35664 |
| CAR_1_60   | mean_car                |  876 |     696 |       23 |   -0.00843 | -0.03773 |   0.02087 |   0.55691 |
| CAR_1_60   | positive_minus_negative |  874 |     695 |       23 |    0.01755 | -0.01121 |   0.04632 |   0.21892 |

## 覆盖与限制

- 使用截至本轮可获得的全部股票名录，包括退市公司，并按事件日上市/退市日期判断资格；仍无法认证历史代码迁移与供应商历史覆盖完整。
- 业绩预告、快报通过普通接口按自然公告日分页重新取得（含周末）；VIP接口无权限。正式财报按匹配股票补齐，同时保留公告日和实际公告日用于重叠事件筛查。来源逐分片留有回执。
- 研报仍含 75 份可能截断的旧月度缓存。扩容时接口进一步返回每天10次额度已用尽，已停止请求；未将被截断月份认证为全量。新增完整分片与旧缓存合并去重，报告缺失仍影响覆盖；未放宽同报告期匹配。
- 行情、换手率、复权因子和利润表的股票级请求均已完成；其中 1 份复权快照为空，请求完成不代表供应商数据无缺失。停牌、缺失、未成熟收益窗口保留缺失，不填零。
- 当日未知公告时刻、历史版本修订、价格指数基准口径等限制仍在，不作市场整体效率或交易获利结论。

复现：`python scripts/run_event_validation.py --expanded`。扩容准备与断点续取见 `scripts/expand_event_data.py`，授权原始数据仅存本地 `data_raw/expansion_20260930/`。`source_inventory.csv`、`provenance.json`、`sample_funnel.csv`和事件级账本记录口径。
