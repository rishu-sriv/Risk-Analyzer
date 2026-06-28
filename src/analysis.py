"""
analysis.py — Core return calculations

This module handles everything related to daily returns.
Every risk metric in the project is built on top of what's computed here.
"""

import pandas as pd
import numpy as np


def calculate_daily_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate daily percentage returns from closing prices.

    Formula: (Today's Price - Yesterday's Price) / Yesterday's Price
    Equivalent to: prices['Close'].pct_change()

    Args:
        prices: DataFrame with a 'Close' column (from yfinance)

    Returns:
        Series of daily returns with NaN dropped (first row has no prior day)
    """
    close = prices["Close"].squeeze()  # flatten MultiIndex columns yfinance sometimes returns
    returns = close.pct_change().dropna()
    returns.name = "Daily Return"
    return returns


def return_summary(returns: pd.Series) -> dict:
    """
    Compute key summary statistics for a return series.

    Args:
        returns: Series of daily returns

    Returns:
        Dict with mean, std, best/worst day, win rate, and day counts
    """
    return {
        "mean_daily_return": returns.mean(),
        "std_daily_return": returns.std(),
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "positive_days": int((returns > 0).sum()),
        "negative_days": int((returns < 0).sum()),
        "win_rate": (returns > 0).mean(),
    }


def print_return_summary(ticker: str, returns: pd.Series) -> None:
    """
    Print a formatted return summary to the terminal.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        returns: Series of daily returns
    """
    stats = return_summary(returns)

    print(f"\n{'='*45}")
    print(f"  Daily Return Summary — {ticker}")
    print(f"{'='*45}")
    print(f"  Period          : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Trading Days    : {len(returns):,}")
    print(f"  Mean Daily Ret  : {stats['mean_daily_return']*100:.4f}%")
    print(f"  Std Dev (Daily) : {stats['std_daily_return']*100:.4f}%")
    print(f"  Best Day        : +{stats['best_day']*100:.2f}%")
    print(f"  Worst Day       : {stats['worst_day']*100:.2f}%")
    print(f"  Positive Days   : {stats['positive_days']:,}")
    print(f"  Negative Days   : {stats['negative_days']:,}")
    print(f"  Win Rate        : {stats['win_rate']*100:.1f}%")
    print(f"{'='*45}\n")