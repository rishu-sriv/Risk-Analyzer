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
)

TICKER  = "AAPL"
START   = "2020-01-01"
END     = "2024-12-31"

# Phase 6 tickers — mix of tech, non-tech, and gold to show contrast
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GLD"]


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
    for ticker in TICKERS:
        prices  = fetch(ticker)
        returns_dict[ticker] = calculate_daily_returns(prices)

    # Build and print correlation matrix
    corr = correlation_matrix(returns_dict)
    print_correlation_matrix(corr)

    # Point out the most and least correlated pair
    import pandas as pd
    # mask diagonal (self-correlations of 1.0)
    masked = corr.where(
        ~pd.DataFrame(
            [[i == j for j in range(len(corr))] for i in range(len(corr))],
            index=corr.index,
            columns=corr.columns
        )
    )
    # find highest and lowest pair
    stack      = masked.stack()
    max_pair   = stack.idxmax()
    min_pair   = stack.idxmin()

    print(f"  Most correlated  : {max_pair[0]} & {max_pair[1]}  →  {corr.loc[max_pair]:.3f}")
    print(f"  Least correlated : {min_pair[0]} & {min_pair[1]}  →  {corr.loc[min_pair]:.3f}\n")


if __name__ == "__main__":
    run_phase_6()