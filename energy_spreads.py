"""
Energy Spreads Analysis: Brent-WTI and the 3-2-1 crack spread
--------------------------------------------------------------
Author: Raghu Karthik (MSc Banking and Finance, King's College London)

What it does
  1. Downloads daily spot prices published by the U.S. Energy Information
     Administration (EIA), via FRED (Federal Reserve Bank of St. Louis):
       WTI crude (Cushing), Brent crude (Europe), NY Harbor gasoline and
       NY Harbor No. 2 heating oil.
  2. Builds two classic trading spreads:
       - Brent-WTI spread ($/bbl): the location/quality spread between the
         two main crude benchmarks.
       - 3-2-1 crack spread ($/bbl): a proxy for US refining margins
         (2 barrels of gasoline + 1 barrel of distillate - 3 barrels of crude).
  3. Analyses them: summary statistics, rolling volatility and correlation,
     z-scores, mean-reversion half-life and (if statsmodels is installed)
     an ADF stationarity test.
  4. Runs a simple, illustrative mean-reversion backtest on the Brent-WTI
     spread, with transaction costs and no look-ahead.
  5. Saves charts, a summary CSV and an auto-generated RESULTS.md.

Usage
  pip install -r requirements.txt
  python energy_spreads.py                 # real data from 2015
  python energy_spreads.py --start 2018-01-01 --entry 2 --exit 0.5
  python energy_spreads.py --demo          # synthetic data, pipeline test only

Disclaimer: educational analysis only. The backtest ignores margin, roll costs,
contract specifications and liquidity, and is not investment advice.
"""

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {
    "WTI": "DCOILWTICO",        # $/bbl, WTI Cushing, Oklahoma
    "BRENT": "DCOILBRENTEU",    # $/bbl, Brent, Europe
    "GASOLINE": "DGASNYH",      # $/gal, NY Harbor conventional regular gasoline
    "HEATING_OIL": "DHOILNYH",  # $/gal, NY Harbor No. 2 heating oil
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
GAL_PER_BBL = 42
TRADING_DAYS = 252
EVENTS = {
    "2020-04-20": "WTI spot price turns negative",
    "2022-02-24": "Russia invades Ukraine",
}


# ---------------------------------------------------------------- data
def load_fred(series_id: str) -> pd.Series:
    """Download one FRED series as a float Series indexed by date."""
    df = pd.read_csv(FRED_URL.format(series_id), na_values=".")
    df.columns = ["date", "value"]  # FRED uses DATE or observation_date
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float).rename(series_id)


def synthetic_data(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Random-walk prices for testing the pipeline only. Never publish these."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    wti = 60 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    spread = np.zeros(n)
    spread[0] = 4.0
    for t in range(1, n):  # mean-reverting Brent-WTI spread
        spread[t] = spread[t - 1] + 0.05 * (4.0 - spread[t - 1]) + rng.normal(0, 0.4)
    brent = wti + spread
    crack = np.zeros(n)
    crack[0] = 22.0
    for t in range(1, n):
        crack[t] = crack[t - 1] + 0.03 * (22.0 - crack[t - 1]) + rng.normal(0, 0.9)
    gasoline = (wti + crack + rng.normal(0, 1.0, n)) / GAL_PER_BBL
    heating_oil = (wti + crack + rng.normal(4, 1.5, n)) / GAL_PER_BBL
    return pd.DataFrame(
        {"WTI": wti, "BRENT": brent, "GASOLINE": gasoline, "HEATING_OIL": heating_oil},
        index=idx,
    )


def load_data(start: str, end: str, demo: bool) -> pd.DataFrame:
    if demo:
        print("DEMO MODE: using synthetic random-walk data (for testing only).")
        return synthetic_data(start, end)
    frames = {}
    for name, sid in SERIES.items():
        print(f"Downloading {name} ({sid}) from FRED...")
        try:
            frames[name] = load_fred(sid)
        except Exception as exc:  # network or format problems
            sys.exit(f"Could not download {sid}: {exc}\n"
                     "Check your internet connection, or try again later.")
    df = pd.concat(frames, axis=1).loc[start:end]
    df = df.dropna()  # keep days where all four prices are published
    if df.empty:
        sys.exit("No overlapping data in the chosen date range.")
    return df


# ------------------------------------------------------------- analytics
def build_spreads(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["BRENT_WTI"] = out["BRENT"] - out["WTI"]
    out["CRACK_321"] = (2 * out["GASOLINE"] * GAL_PER_BBL
                        + 1 * out["HEATING_OIL"] * GAL_PER_BBL
                        - 3 * out["WTI"]) / 3
    # Log returns only where prices are positive (WTI was negative in Apr 2020)
    for col in ["WTI", "BRENT"]:
        px = out[col].where(out[col] > 0)
        out[f"{col}_RET"] = np.log(px).diff()
        out[f"{col}_VOL"] = out[f"{col}_RET"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    out["CORR"] = out["WTI_RET"].rolling(window).corr(out["BRENT_RET"])
    s = out["BRENT_WTI"]
    out["SPREAD_MEAN"] = s.rolling(window).mean()
    out["SPREAD_STD"] = s.rolling(window).std()
    out["Z"] = (s - out["SPREAD_MEAN"]) / out["SPREAD_STD"]
    return out


def half_life(series: pd.Series) -> float:
    """Mean-reversion half-life in days from an AR(1) regression of the change on the lagged level."""
    s = series.dropna()
    lagged, delta = s.shift(1).iloc[1:], s.diff().iloc[1:]
    beta = np.polyfit(lagged.values, delta.values, 1)[0]
    return float(-np.log(2) / beta) if beta < 0 else float("nan")


def adf_test(series: pd.Series):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
    stat, pvalue, *_ = adfuller(series.dropna(), autolag="AIC")
    return stat, pvalue


def summary_stats(data: pd.DataFrame) -> pd.DataFrame:
    last_year = data.loc[data.index.max() - pd.DateOffset(years=1):]
    rows = {}
    for col, label in [("BRENT_WTI", "Brent-WTI spread ($/bbl)"),
                       ("CRACK_321", "3-2-1 crack spread ($/bbl)")]:
        rows[label] = {
            "Mean (full period)": data[col].mean(),
            "Std dev (full period)": data[col].std(),
            "Min": data[col].min(),
            "Max": data[col].max(),
            "Mean (last 12 months)": last_year[col].mean(),
            "Latest": data[col].iloc[-1],
        }
    return pd.DataFrame(rows).T.round(2)


def backtest(data: pd.DataFrame, entry: float, exit_z: float, cost: float) -> dict:
    """Mean reversion on the Brent-WTI z-score.
    z > entry: short the spread (sell Brent, buy WTI); z < -entry: long the spread.
    Close when |z| < exit_z. Signals use information up to day t; P&L starts on t+1."""
    z = data["Z"].values
    pos = np.zeros(len(z))
    for t in range(1, len(z)):
        prev = pos[t - 1]
        if np.isnan(z[t]):
            pos[t] = prev
        elif prev == 0:
            pos[t] = -1 if z[t] > entry else (1 if z[t] < -entry else 0)
        elif abs(z[t]) < exit_z:
            pos[t] = 0
        else:
            pos[t] = prev
    position = pd.Series(pos, index=data.index)
    gross = position.shift(1).fillna(0) * data["BRENT_WTI"].diff().fillna(0)
    costs = cost * position.diff().abs().fillna(0)
    pnl = gross - costs
    equity = pnl.cumsum()

    # per-trade P&L
    trades, current, open_ = [], 0.0, False
    for t in range(1, len(pos)):
        if pos[t - 1] != 0:
            current += pnl.iloc[t]
        if pos[t - 1] == 0 and pos[t] != 0:          # open
            open_, current = True, -costs.iloc[t]
        elif open_ and pos[t] == 0:                   # close
            trades.append(current)
            open_, current = False, 0.0
    drawdown = equity - equity.cummax()
    daily_std = pnl.std()
    return {
        "position": position,
        "equity": equity,
        "n_trades": len(trades),
        "hit_rate": (np.mean([x > 0 for x in trades]) if trades else float("nan")),
        "total_pnl": float(equity.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "sharpe_proxy": (float(pnl.mean() / daily_std * np.sqrt(TRADING_DAYS))
                         if daily_std > 0 else float("nan")),
        "time_in_market": float((position != 0).mean()),
    }


# ---------------------------------------------------------------- charts
def _events(ax, data):
    for d, label in EVENTS.items():
        ts = pd.Timestamp(d)
        if data.index.min() <= ts <= data.index.max():
            ax.axvline(ts, color="grey", linestyle="--", linewidth=0.8)
            ax.annotate(label, (ts, ax.get_ylim()[1]), rotation=90, va="top",
                        ha="right", fontsize=7, color="grey")


def make_charts(data: pd.DataFrame, bt: dict, chart_dir: str, demo: bool) -> list:
    os.makedirs(chart_dir, exist_ok=True)
    tag = " (SYNTHETIC DEMO DATA)" if demo else ""
    plt.rcParams.update({"figure.figsize": (10, 4.5), "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    paths = []

    fig, ax = plt.subplots()
    ax.plot(data.index, data["WTI"], label="WTI", linewidth=1)
    ax.plot(data.index, data["BRENT"], label="Brent", linewidth=1)
    ax.set_title("Crude oil spot prices ($/bbl)" + tag)
    ax.legend(); _events(ax, data)
    p = os.path.join(chart_dir, "01_crude_prices.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(data.index, data["BRENT_WTI"], linewidth=0.9, label="Brent-WTI")
    ax.plot(data.index, data["SPREAD_MEAN"], linewidth=1.2, label="Rolling mean")
    ax.fill_between(data.index, data["SPREAD_MEAN"] - 2 * data["SPREAD_STD"],
                    data["SPREAD_MEAN"] + 2 * data["SPREAD_STD"], alpha=0.15,
                    label="±2 rolling std")
    ax.set_title("Brent-WTI spread ($/bbl)" + tag); ax.legend(); _events(ax, data)
    p = os.path.join(chart_dir, "02_brent_wti_spread.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(data.index, data["CRACK_321"], linewidth=0.9, color="tab:green")
    ax.set_title("3-2-1 crack spread, a proxy for US refining margins ($/bbl)" + tag)
    _events(ax, data)
    p = os.path.join(chart_dir, "03_crack_spread.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax1 = plt.subplots()
    ax1.plot(data.index, data["WTI_VOL"], linewidth=0.9, label="WTI vol (annualised)")
    ax1.plot(data.index, data["BRENT_VOL"], linewidth=0.9, label="Brent vol (annualised)")
    ax1.set_ylabel("Volatility"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx(); ax2.plot(data.index, data["CORR"], color="black", linewidth=0.7, alpha=0.6)
    ax2.set_ylabel("Rolling correlation (black)"); ax2.set_ylim(-1, 1.05)
    ax1.set_title("Rolling volatility and Brent-WTI return correlation" + tag)
    p = os.path.join(chart_dir, "04_vol_and_correlation.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(bt["equity"].index, bt["equity"], linewidth=1)
    ax.set_title("Illustrative mean-reversion backtest on Brent-WTI: cumulative P&L ($/bbl)" + tag)
    p = os.path.join(chart_dir, "05_backtest_equity.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------- report
def write_results(data, stats, hl_spread, hl_crack, adf, bt, args, out_dir, charts):
    lines = []
    if args.demo:
        lines.append("> **WARNING: these results use SYNTHETIC demo data. Run without --demo before sharing.**\n")
    lines += [
        "# Results: Brent-WTI and 3-2-1 crack spread analysis",
        "",
        f"Data window: {data.index.min():%d %b %Y} to {data.index.max():%d %b %Y} "
        f"({len(data):,} trading days). Source: EIA spot prices via FRED.",
        "",
        "## Summary statistics",
        "",
        stats.to_markdown(),
        "",
        "## Mean reversion",
        "",
        f"- Brent-WTI half-life: **{hl_spread:.1f} trading days**",
        f"- 3-2-1 crack spread half-life: **{hl_crack:.1f} trading days**",
    ]
    if adf:
        lines.append(f"- ADF test on Brent-WTI: statistic {adf[0]:.2f}, p-value {adf[1]:.3f} "
                     f"({'stationary at 5%' if adf[1] < 0.05 else 'not stationary at 5%'})")
    else:
        lines.append("- ADF test skipped (install statsmodels to run it).")
    lines += [
        f"- Latest Brent-WTI z-score ({args.window}-day window): **{data['Z'].iloc[-1]:.2f}**",
        "",
        "## Illustrative backtest (Brent-WTI z-score mean reversion)",
        "",
        f"Rules: enter when |z| > {args.entry}, exit when |z| < {args.exit}; "
        f"cost {args.cost} $/bbl per position change; P&L per 1 bbl of spread.",
        "",
        f"- Trades: **{bt['n_trades']}**",
        f"- Hit rate: **{bt['hit_rate']:.0%}**" if bt['n_trades'] else "- Hit rate: n/a",
        f"- Total P&L: **{bt['total_pnl']:.2f} $/bbl**",
        f"- Maximum drawdown: **{bt['max_drawdown']:.2f} $/bbl**",
        f"- Sharpe proxy (annualised): **{bt['sharpe_proxy']:.2f}**",
        f"- Time in market: **{bt['time_in_market']:.0%}**",
        "",
        "Caveat: spot-price backtest only. A real trade would use futures, face roll costs, "
        "margin and liquidity constraints, and different contract specifications.",
        "",
        "## Charts",
        "",
    ]
    lines += [f"![{os.path.basename(c)}](charts/{os.path.basename(c)})" for c in charts]
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Brent-WTI and 3-2-1 crack spread analysis")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--window", type=int, default=60, help="rolling window in trading days")
    ap.add_argument("--entry", type=float, default=2.0, help="z-score to open a trade")
    ap.add_argument("--exit", type=float, default=0.5, help="z-score to close a trade")
    ap.add_argument("--cost", type=float, default=0.05, help="cost per position change, $/bbl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--demo", action="store_true", help="synthetic data, for testing only")
    args = ap.parse_args()

    raw = load_data(args.start, args.end, args.demo)
    data = build_spreads(raw, args.window)
    stats = summary_stats(data)
    hl_spread, hl_crack = half_life(data["BRENT_WTI"]), half_life(data["CRACK_321"])
    adf = adf_test(data["BRENT_WTI"])
    bt = backtest(data, args.entry, args.exit, args.cost)

    os.makedirs(args.out, exist_ok=True)
    charts = make_charts(data, bt, os.path.join(args.out, "charts"), args.demo)
    stats.to_csv(os.path.join(args.out, "summary_stats.csv"))
    data.to_csv(os.path.join(args.out, "spreads_daily.csv"))
    results = write_results(data, stats, hl_spread, hl_crack, adf, bt, args, args.out, charts)

    print("\n" + stats.to_string())
    print(f"\nHalf-life Brent-WTI: {hl_spread:.1f} days | crack: {hl_crack:.1f} days")
    print(f"Backtest: {bt['n_trades']} trades, total P&L {bt['total_pnl']:.2f} $/bbl, "
          f"max drawdown {bt['max_drawdown']:.2f}")
    print(f"\nSaved charts, CSVs and {results}")


if __name__ == "__main__":
    main()
