# Energy Spreads Analysis: Brent–WTI and the 3-2-1 Crack Spread

A Python project analysing two spreads that energy traders watch every day, using public daily price data from the U.S. Energy Information Administration (EIA).

**Author:** Raghu Karthik, MSc Banking and Finance, King's College London

## Why these spreads matter

- **Brent–WTI ($/bbl)** is the price gap between the two main crude benchmarks. WTI is priced at Cushing, Oklahoma, and is landlocked. Brent is waterborne North Sea crude. The gap reflects transport and export capacity, crude quality and regional supply and demand, so it is a core relative-value and hedging spread.
- **The 3-2-1 crack spread ($/bbl)** approximates US refining margins: the value of 2 barrels of gasoline plus 1 barrel of distillate (heating oil), minus 3 barrels of crude. Refiners and traders use it to track and hedge refining profitability.

## What the project does

1. Downloads EIA daily spot prices via FRED (no API key needed):

   | Series | FRED ID | Unit |
   |---|---|---|
   | WTI crude, Cushing | `DCOILWTICO` | $/bbl |
   | Brent crude, Europe | `DCOILBRENTEU` | $/bbl |
   | NY Harbor conventional gasoline | `DGASNYH` | $/gal |
   | NY Harbor No. 2 heating oil | `DHOILNYH` | $/gal |

2. Builds both spreads (product prices converted at 42 gallons per barrel).
3. Analyses them:
   - summary statistics, full period and last 12 months
   - rolling 60-day annualised volatility and Brent–WTI return correlation
   - rolling z-scores
   - mean-reversion **half-life** from an AR(1) regression of daily changes on lagged levels
   - an optional **ADF stationarity test** (needs `statsmodels`)
4. Runs an **illustrative mean-reversion backtest** on Brent–WTI. It opens a trade when the z-score moves beyond ±2 and closes it inside ±0.5, with transaction costs and no look-ahead.
5. Saves five charts, two CSV files and an auto-generated `RESULTS.md` containing the numbers from your run.

## How to run

```bash
pip install -r requirements.txt
python energy_spreads.py                   # real data, 2015 to today
python energy_spreads.py --start 2018-01-01 --entry 2 --exit 0.5 --cost 0.05
python energy_spreads.py --demo            # synthetic data: tests the pipeline only
```

Outputs are written to `outputs/`: `charts/*.png`, `summary_stats.csv`, `spreads_daily.csv` and `RESULTS.md`.

## Results



Key findings (EIA data, 2 January 2015 to 27 October 2026, 2,912 trading days):
- **2026 is exceptional for Brent–WTI.** The spread has averaged **$9.99/bbl in 2026**, against **$4.05/bbl over 2015–2025**, and peaked at **$37.62/bbl on 2 October 2026** (latest: $19.73). That pattern fits a supply shock that hits seaborne crude harder than landlocked US barrels.
- **Refining margins set a new high.** The 3-2-1 crack spread reached **$78.92/bbl on 25 September 2026**, above its 2022 peak of $71.74 (16 May 2022). It has averaged **$48.63 in 2026**, against $16.32 over 2015–2021.
- **Brent–WTI mean-reverts quickly.** The half-life is about **5.7 trading days**, and the ADF test rejects a unit root (p = 0.002). The crack spread is far more persistent, with a half-life of about **44 trading days**.
- **The backtest has a high hit rate but fat tails.** The z-score rule made **80 trades with a 95% hit rate** and **$204.59/bbl** of cumulative P&L. However, **one day, 21 April 2020, contributed $54.08 (26%)**. Excluding March–May 2020, P&L falls to $141.30/bbl over 78 trades, with a Sharpe proxy of 0.82 and a maximum drawdown of −$22.57/bbl.
- **Correlation has broken down.** The 60-day correlation of Brent and WTI daily returns fell to **0.43** by late October 2026, well below its usual 0.8–0.95 range.
- The full-period extremes for Brent–WTI (maximum $54.34, minimum −$5.66) are distorted by WTI spot turning negative in April 2020.

## Limitations

- These are spot prices. A real trade would use futures (ICE Brent, NYMEX WTI, RBOB, ULSD), with roll costs, margin, contract sizes and liquidity limits.
- The NY Harbor products are a regional proxy, and the 3-2-1 ratio is a simplification of real refinery yields.
- The backtest is for learning only. It is in-sample, ignores financing and slippage, and is not investment advice.

## Next steps

- Use futures curves to study term structure (contango and backwardation) and calendar spreads.
- Add EIA weekly inventory data and test its effect on spreads around release days.
- Try regime detection or simple machine-learning forecasts, using walk-forward validation.

## Tools

Python (pandas, NumPy, Matplotlib; statsmodels optional). Data: U.S. EIA via FRED, Federal Reserve Bank of St. Louis.
