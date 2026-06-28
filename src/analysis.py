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


# ── Correlation ───────────────────────────────────────────────────────────────

def build_returns_dataframe(returns_dict: dict) -> "pd.DataFrame":
    """
    Combine multiple return series into one aligned DataFrame.

    Each key becomes a column. Dates that don't exist for all stocks
    are dropped (inner join) so every row has data for every ticker.

    Args:
        returns_dict: Dict of {ticker: pd.Series of daily returns}

    Returns:
        DataFrame with one column per ticker, aligned on date
    """
    import pandas as pd
    df = pd.DataFrame(returns_dict)
    df = df.dropna()   # drop any dates missing for any ticker
    return df


def correlation_matrix(returns_dict: dict) -> "pd.DataFrame":
    """
    Compute pairwise Pearson correlation between all stocks.

    Result is a symmetric NxN matrix where N = number of tickers.
    Diagonal is always 1.0 (stock perfectly correlated with itself).
    Off-diagonal shows how similarly each pair moves day to day.

    Args:
        returns_dict: Dict of {ticker: pd.Series of daily returns}

    Returns:
        DataFrame — the correlation matrix
    """
    df = build_returns_dataframe(returns_dict)
    return df.corr()


def print_correlation_matrix(corr_matrix: "pd.DataFrame") -> None:
    """
    Print the correlation matrix in a readable format.

    Args:
        corr_matrix: DataFrame output from correlation_matrix()
    """
    import pandas as pd
    tickers = corr_matrix.columns.tolist()
    col_w   = 10

    print(f"\n{'='*55}")
    print(f"  Correlation Matrix")
    print(f"{'='*55}")

    # header row
    header = f"  {'':8}"
    for t in tickers:
        header += f"{t:>{col_w}}"
    print(header)
    print(f"  {'-'*( 8 + col_w * len(tickers))}")

    # data rows
    for row_ticker in tickers:
        row = f"  {row_ticker:<8}"
        for col_ticker in tickers:
            val = corr_matrix.loc[row_ticker, col_ticker]
            # highlight diagonal and strong correlations
            row += f"{val:>{col_w}.3f}"
        print(row)

    print(f"{'='*55}")
    print(f"  1.000 = perfect positive  |  0.000 = no relation")
    print(f"  Negative = move in opposite directions\n")