"""
main.py — Terminal runner

Run this file to test each phase as you build it.
Change TICKER and date range to explore different stocks.
"""

import yfinance as yf
from src.analysis import calculate_daily_returns, print_return_summary
from src.risk_metrics import (
    print_volatility_summary,
    print_drawdown_summary,
    print_distribution_summary,
    annualized_volatility,
)

TICKER = "AAPL"
START  = "2020-01-01"
END    = "2024-12-31"


def fetch(ticker: str):
    print(f"\nFetching data for {ticker}...")
    prices = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if prices.empty:
        raise ValueError(f"No data found for {ticker}. Check the ticker symbol.")
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
        vol = annualized_volatility(returns)
        print(f"  {ticker:<6} annualized vol: {vol*100:.2f}%")
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

    # Compare distribution shape across two different stocks
    print("\n── Distribution comparison: AAPL vs GME ──")
    from src.risk_metrics import distribution_stats
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


if __name__ == "__main__":
    run_phase_5()