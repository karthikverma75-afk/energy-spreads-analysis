# Results: Brent-WTI and 3-2-1 crack spread analysis

Data window: 02 Jan 2015 to 27 Oct 2026 (2,912 trading days). Source: EIA spot prices via FRED.

## Summary statistics

|                            |   Mean (full period) |   Std dev (full period) |   Min |   Max |   Mean (last 12 months) |   Latest |
|:---------------------------|---------------------:|------------------------:|------:|------:|------------------------:|---------:|
| Brent-WTI spread ($/bbl)   |                 4.46 |                    3.85 | -5.66 | 54.34 |                    8.93 |    19.73 |
| 3-2-1 crack spread ($/bbl) |                22.83 |                   12.42 |  1.7  | 78.92 |                   44.29 |    40.9  |

## Mean reversion

- Brent-WTI half-life: **5.7 trading days**
- 3-2-1 crack spread half-life: **44.3 trading days**
- ADF test on Brent-WTI: statistic -3.98, p-value 0.002 (stationary at 5%)
- Latest Brent-WTI z-score (60-day window): **0.41**

## Illustrative backtest (Brent-WTI z-score mean reversion)

Rules: enter when |z| > 2.0, exit when |z| < 0.5; cost 0.05 $/bbl per position change; P&L per 1 bbl of spread.

- Trades: **80**
- Hit rate: **95%**
- Total P&L: **204.59 $/bbl**
- Maximum drawdown: **-22.57 $/bbl**
- Sharpe proxy (annualised): **0.80**
- Time in market: **34%**

Caveat: spot-price backtest only. A real trade would use futures, face roll costs, margin and liquidity constraints, and different contract specifications.

## Charts

![01_crude_prices.png](charts/01_crude_prices.png)
![02_brent_wti_spread.png](charts/02_brent_wti_spread.png)
![03_crack_spread.png](charts/03_crack_spread.png)
![04_vol_and_correlation.png](charts/04_vol_and_correlation.png)
![05_backtest_equity.png](charts/05_backtest_equity.png)
