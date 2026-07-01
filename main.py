"""
main.py — Terminal runner

Run this file to test each phase as you build it.
Change TICKER and date range to explore different stocks.
"""

import yfinance as yf
from src.analysis import (
    calculate_daily_returns,
    print_return_summary,
    correlation_matrix,
    print_correlation_matrix,
)
from src.risk_metrics import (
    print_volatility_summary,
    print_drawdown_summary,
    print_distribution_summary,
    annualized_volatility,
    distribution_stats,
    build_comparison_table,
    print_comparison_table,
)

TICKER  = "AAPL"
START   = "2020-01-01"
END     = "2024-12-31"

# Phase 7 — diverse mix: tech, finance, consumer, gold, volatile
COMPARE_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "GLD", "GME"]


def fetch(ticker: str):
    print(f"  Fetching {ticker}...")
    prices = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if prices.empty:
        raise ValueError(f"No data found for {ticker}.")
    return prices


def run_phase_2():
    prices  = fetch(TICKER)
    returns = calculate_daily_returns(prices)
    print_return_summary(TICKER, returns)


def run_phase_3():
    prices  = fetch(TICKER)
    returns = calculate_daily_returns(prices)
    print_return_summary(TICKER, returns)
    print_volatility_summary(TICKER, returns)

    print("\n── Volatility comparison: AAPL vs JNJ ──")
    for ticker in ["AAPL", "JNJ"]:
        prices  = fetch(ticker)
        returns = calculate_daily_returns(prices)
        print(f"  {ticker:<6} annualized vol: {annualized_volatility(returns)*100:.2f}%")
    print()


def run_phase_4():
    prices  = fetch(TICKER)
    returns = calculate_daily_returns(prices)
    print_return_summary(TICKER, returns)
    print_volatility_summary(TICKER, returns)
    print_drawdown_summary(TICKER, returns)


def run_phase_5():
    prices  = fetch(TICKER)
    returns = calculate_daily_returns(prices)
    print_return_summary(TICKER, returns)
    print_drawdown_summary(TICKER, returns)
    print_distribution_summary(TICKER, returns)

    print("\n── Distribution comparison: AAPL vs GME ──")
    for ticker in ["AAPL", "GME"]:
        prices  = fetch(ticker)
        returns = calculate_daily_returns(prices)
        s = distribution_stats(returns)
        print(f"\n  {ticker}")
        print(f"    Skewness  : {s['skewness']:.3f}")
        print(f"    Kurtosis  : {s['kurtosis']:.3f}")
        print(f"    VaR (95%) : {s['var_95']*100:.2f}%")
        print(f"    Win Rate  : {s['win_rate']*100:.1f}%")
    print()


def run_phase_6():
    print(f"\nFetching data for correlation analysis...")
    returns_dict = {}
    for ticker in ["AAPL", "MSFT", "GOOGL", "JPM", "GLD"]:
        prices = fetch(ticker)
        returns_dict[ticker] = calculate_daily_returns(prices)

    corr = correlation_matrix(returns_dict)
    print_correlation_matrix(corr)

    import pandas as pd
    masked  = corr.where(~pd.DataFrame(
        [[i == j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns
    ))
    stack     = masked.stack()
    max_pair  = stack.idxmax()
    min_pair  = stack.idxmin()
    print(f"  Most correlated  : {max_pair[0]} & {max_pair[1]}  →  {corr.loc[max_pair]:.3f}")
    print(f"  Least correlated : {min_pair[0]} & {min_pair[1]}  →  {corr.loc[min_pair]:.3f}\n")


def run_phase_7():
    print(f"\nFetching data for {len(COMPARE_TICKERS)} stocks...\n")

    tickers_returns = {}
    for ticker in COMPARE_TICKERS:
        prices = fetch(ticker)
        tickers_returns[ticker] = calculate_daily_returns(prices)

    # Build and print the full risk comparison table
    df = build_comparison_table(tickers_returns)
    print_comparison_table(df)

    # Highlight key takeaways
    best_sharpe  = df["Sharpe Ratio"].idxmax()
    safest       = df["Max Drawdown"].idxmax()   # closest to 0 = least drawdown
    most_volatile= df["Ann. Volatility"].idxmax()

    print(f"  Best risk-adjusted return : {best_sharpe}  (Sharpe {df.loc[best_sharpe, 'Sharpe Ratio']:.3f})")
    print(f"  Safest (least drawdown)   : {safest}  (Max DD {df.loc[safest, 'Max Drawdown']*100:.2f}%)")
    print(f"  Most volatile             : {most_volatile}  (Vol {df.loc[most_volatile, 'Ann. Volatility']*100:.2f}%)\n")


if __name__ == "__main__":
    run_phase_7()