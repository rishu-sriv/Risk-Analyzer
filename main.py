"""
main.py — Terminal runner for Phase 2

Run this file to test your daily return calculations.
Change the ticker and date range to explore different stocks.
"""

import yfinance as yf
from src.analysis import calculate_daily_returns, print_return_summary


def run_phase_2():
    TICKER = "AAPL"
    START = "2020-01-01"
    END = "2024-12-31"

    print(f"\nFetching data for {TICKER}...")
    prices = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

    if prices.empty:
        print(f"No data found for {TICKER}. Check the ticker symbol.")
        return

    returns = calculate_daily_returns(prices)

    # Print formatted summary
    print_return_summary(TICKER, returns)

    # Print the raw describe() — read every row carefully
    print("describe() output — raw pandas statistics:")
    print(returns.describe().apply(lambda x: f"{x*100:.4f}%"))

    # Print a few return values so you can see what the Series looks like
    print(f"\nFirst 5 daily returns:")
    print(returns.head().apply(lambda x: f"{x*100:.4f}%"))

    print(f"\nLast 5 daily returns:")
    print(returns.tail().apply(lambda x: f"{x*100:.4f}%"))


if __name__ == "__main__":
    run_phase_2()