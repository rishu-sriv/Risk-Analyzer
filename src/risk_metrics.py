"""
risk_metrics.py — Core risk metric calculations

This is the heart of the Risk Analyzer.
Every function here answers a different question about risk.

Phase 3: Volatility (daily, annualized, rolling)
Phase 4: Drawdown (coming next)
Phase 5: Return distribution (coming next)
"""

import pandas as pd
import numpy as np

TRADING_DAYS = 252  # approximate trading days in a year


# ── Volatility ──────────────────────────────────────────────────────────────

def daily_volatility(returns: pd.Series) -> float:
    """
    Standard deviation of daily returns.
    This is the raw, unscaled volatility number.

    Args:
        returns: Series of daily returns

    Returns:
        Float — daily std dev (e.g. 0.0199 means ~2% per day)
    """
    return returns.std()


def annualized_volatility(returns: pd.Series) -> float:
    """
    Volatility scaled to annual.
    Formula: daily_std × √252

    Why √252 and not ×252?
    Variance scales linearly with time.
    Std dev (volatility) scales with the square root of time.
    So: annual_std = daily_std × √(trading days per year)

    Args:
        returns: Series of daily returns

    Returns:
        Float — annualized volatility (e.g. 0.317 means ~31.7% per year)
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Annualized rolling volatility over a sliding window.

    Instead of one flat number across all time, this shows how volatility
    changed over time — spikes during crashes, calm during bull markets.

    Args:
        returns: Series of daily returns
        window: Number of trading days to look back (30 = ~1 month, 90 = ~1 quarter)

    Returns:
        Series of annualized rolling volatility values
        First (window - 1) rows will be NaN — not enough history yet
    """
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)


def volatility_summary(returns: pd.Series) -> dict:
    """
    All three volatility metrics in one dict.

    Args:
        returns: Series of daily returns

    Returns:
        Dict with daily vol, annualized vol, and rolling vol series (30d and 90d)
    """
    return {
        "daily_volatility": daily_volatility(returns),
        "annualized_volatility": annualized_volatility(returns),
        "rolling_30d": rolling_volatility(returns, window=30),
        "rolling_90d": rolling_volatility(returns, window=90),
    }


def print_volatility_summary(ticker: str, returns: pd.Series) -> None:
    """
    Print a formatted volatility summary to the terminal.

    Args:
        ticker: Stock ticker symbol
        returns: Series of daily returns
    """
    daily_vol = daily_volatility(returns)
    annual_vol = annualized_volatility(returns)
    roll_30 = rolling_volatility(returns, window=30)
    roll_90 = rolling_volatility(returns, window=90)

    print(f"\n{'='*45}")
    print(f"  Volatility Summary — {ticker}")
    print(f"{'='*45}")
    print(f"  Daily Volatility      : {daily_vol*100:.4f}%")
    print(f"  Annualized Volatility : {annual_vol*100:.2f}%")
    print(f"  ---")
    print(f"  Rolling 30d Vol (cur) : {roll_30.iloc[-1]*100:.2f}%")
    print(f"  Rolling 30d Vol (max) : {roll_30.max()*100:.2f}%")
    print(f"  Rolling 30d Vol (min) : {roll_30.min()*100:.2f}%")
    print(f"  ---")
    print(f"  Rolling 90d Vol (cur) : {roll_90.iloc[-1]*100:.2f}%")
    print(f"  Rolling 90d Vol (max) : {roll_90.max()*100:.2f}%")
    print(f"  Rolling 90d Vol (min) : {roll_90.min()*100:.2f}%")
    print(f"{'='*45}\n")


# ── Drawdown ─────────────────────────────────────────────────────────────────

def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate the drawdown series from daily returns.

    At each point in time, drawdown = % fall from the highest peak reached so far.
    When the stock is at an all-time high, drawdown = 0%.
    When it has fallen from its peak, drawdown is negative.

    How it works:
        1. Convert daily returns → cumulative growth curve  (1+r1)(1+r2)...
        2. Track the running peak of that curve with cummax()
        3. Drawdown = (current - peak) / peak

    Args:
        returns: Series of daily returns

    Returns:
        Series of drawdown values (always <= 0)
    """
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return drawdown


def max_drawdown(returns: pd.Series) -> float:
    """
    The single worst drawdown over the entire period.
    This is the minimum value of the drawdown series.

    A result of -0.35 means the stock fell 35% from its peak at worst.

    Args:
        returns: Series of daily returns

    Returns:
        Float — maximum drawdown (always negative, e.g. -0.35 = -35%)
    """
    return calculate_drawdown(returns).min()


def drawdown_duration(returns: pd.Series) -> dict:
    """
    Find the peak and trough dates of the maximum drawdown,
    and calculate how many trading days the drawdown lasted.

    Args:
        returns: Series of daily returns

    Returns:
        Dict with peak_date, trough_date, and duration_days
    """
    drawdown    = calculate_drawdown(returns)
    cumulative  = (1 + returns).cumprod()

    trough_date = drawdown.idxmin()                          # date of worst drawdown
    peak_date   = cumulative.loc[:trough_date].idxmax()      # date of prior peak

    # count trading days between peak and trough (calendar days can be misleading)
    trading_days = len(returns.loc[peak_date:trough_date])

    return {
        "peak_date"    : peak_date,
        "trough_date"  : trough_date,
        "duration_days": trading_days,
    }


def print_drawdown_summary(ticker: str, returns: pd.Series) -> None:
    """
    Print a formatted drawdown summary to the terminal.

    Args:
        ticker: Stock ticker symbol
        returns: Series of daily returns
    """
    mdd      = max_drawdown(returns)
    duration = drawdown_duration(returns)
    dd_now   = calculate_drawdown(returns).iloc[-1]

    print(f"\n{'='*45}")
    print(f"  Drawdown Summary — {ticker}")
    print(f"{'='*45}")
    print(f"  Max Drawdown           : {mdd*100:.2f}%")
    print(f"  Peak Date              : {duration['peak_date'].date()}")
    print(f"  Trough Date            : {duration['trough_date'].date()}")
    print(f"  Duration (trading days): {duration['duration_days']}")
    print(f"  Current Drawdown       : {dd_now*100:.2f}%")
    print(f"{'='*45}\n")