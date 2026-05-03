# Data License

This project uses financial data from **Tushare Pro** (https://tushare.pro/) and **AkShare** (https://akshare.xyz/).

## Redistribution Policy
- **Raw Data**: Raw financial data downloaded via Tushare or AkShare is **NOT** redistributed in this repository in compliance with their terms of service.
- **Processed Data**: Intermediate processed datasets provided in this repository are for demonstration and diagnostic purposes only. They should not be used for commercial trading.

## Requirements for Users
To reproduce the full analysis, users must:
1. Obtain their own **Tushare Pro** token.
2. Be aware that Tushare permission tiers (积分系统) may affect the availability of certain endpoints (like `forecast_vip`, `express_vip`, or certain analyst coverage reports), which will in turn affect the final sample size and coverage statistics.
3. Generated outputs may differ slightly depending on the exact date of the API update and the user's specific access rights.

## Disclaimer
The authors of this project are not responsible for any financial losses incurred through the use of this code or data.
