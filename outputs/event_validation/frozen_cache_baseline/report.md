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
| unique_forecast_rows                       |    3069 |
| initial_forecasts_in_price_span            |    2992 |
| legacy_next_session_cutoff_matched         |     344 |
| legacy_latest_report_on_after_announcement |      67 |
| strict_announcement_cutoff_matched         |     344 |
| prior_liquidity_and_listing_eligible       |     159 |
| complete_adjusted_CAR_1_10                 |     159 |
| complete_adjusted_CAR_1_10_no_overlap      |     157 |

报告期覆盖：

| period_label   |   events |   matched |   match_rate |
|:---------------|---------:|----------:|-------------:|
| 03-31          |      494 |         0 |     0        |
| 06-30          |     1018 |         0 |     0        |
| 09-30          |      412 |         0 |     0        |
| 12-31          |     1068 |       344 |     0.322097 |

## 重算结果

以下均为算术累计市场调整收益、小数单位；例如 0.01 为 1%。CI 为 95% 双向 firm/month 聚类区间。

| window     | statistic               |   n |   firms |   months |   estimate |    ci_low |   ci_high |   p_value |
|:-----------|:------------------------|----:|--------:|---------:|-----------:|----------:|----------:|----------:|
| CAR_-10_-1 | mean_car                | 155 |      63 |       14 |   -0.00465 |  -0.02884 |   0.01953 |   0.68435 |
| CAR_-10_-1 | positive_minus_negative | 155 |      63 |       14 |    0.01121 |  -0.02322 |   0.04564 |   0.49421 |
| CAR_0_1    | mean_car                | 157 |      63 |       14 |   -0.00173 |  -0.01332 |   0.00985 |   0.75185 |
| CAR_0_1    | positive_minus_negative | 157 |      63 |       14 |    0.00412 |  -0.00606 |   0.01430 |   0.39818 |
| CAR_1_10   | mean_car                | 157 |      63 |       14 |    0.01584 |  -0.00298 |   0.03467 |   0.09209 |
| CAR_1_10   | positive_minus_negative | 157 |      63 |       14 |   -0.01314 |  -0.05109 |   0.02481 |   0.46785 |
| CAR_1_20   | mean_car                | 155 |      62 |       13 |    0.03547 |   0.00952 |   0.06142 |   0.01152 |
| CAR_1_20   | positive_minus_negative | 155 |      62 |       13 |   -0.01358 |  -0.05345 |   0.02629 |   0.47234 |
| CAR_1_60   | mean_car                |  14 |      12 |        7 |    0.04433 | nan       | nan       | nan       |
| CAR_1_60   | positive_minus_negative |  14 |      12 |        7 |    0.04431 | nan       | nan       | nan       |

主检验 N=157，差值=-0.01314，CI=[-0.05109, 0.02481]，p=0.46785。
这里检验的是盈利意外与后续收益的关联，不是总样本平均收益是否为正；平均 CAR 不能证明 PEAD。

## 仍然限制结论的来源缺口

- 76 个月的历史 report_rc 文件疑似触及接口上限；不能据此宣称覆盖整个市场。新 loader 已修分页，但本轮没有把缺失历史研报补造出来。需完整重抓并保留当时可得版本后再确认结果。
- 股票池是当前上市股票代码排序前 300 名，存在幸存者与市场板块偏差；不代表全部 A 股。
- 公告仅有日期，采用次交易日对齐的保守约定；无盘前/盘后时间戳。快报/正式财报缓存可能遗漏后续事件，污染剔除只是已观测事件层面。
- 价格指数基准与含公司行动的个股收益口径不完全对称；双向聚类不能消除这些测量偏差。尚未验证行业/因子基准，不作市场效率或交易获利结论。
- 冻结阈值后的历史回看不等于真正前瞻样本外；180/365 天和 1/2 机构结果全部保留，不选显著项。
- 旧 PDF、演示稿和 outputs/tables 是历史产物，未同步重编；本目录是当前审计版本。

## 复现与文件

`python scripts/run_event_validation.py` 完全离线使用已保存复权因子；首次补因子用 `--refresh-factors`。
`events.csv` 是事件级账本；`matching_audit.csv` 保留失配行与修复前后预期；`sample_funnel.csv`、`period_coverage.csv`、`source_inventory.csv`、`inference.csv` 和 `provenance.json` 可逐项复核。

字段定义：[report_rc](https://tushare.pro/document/2?doc_id=292)、[forecast](https://tushare.pro/document/2?doc_id=45)、[express](https://tushare.pro/document/2?doc_id=46)、[fina_indicator](https://tushare.pro/document/2?doc_id=79)。
