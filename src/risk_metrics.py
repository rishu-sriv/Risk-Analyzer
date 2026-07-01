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


# ── Return Distribution ───────────────────────────────────────────────────────

def distribution_stats(returns: pd.Series) -> dict:
    """
    Statistical shape of the return distribution.

    Goes beyond mean and std to describe *how* returns are distributed —
    are bad days worse than good days are good? Do extreme events happen
    more often than a normal distribution would predict?

    Args:
        returns: Series of daily returns

    Returns:
        Dict with skewness, kurtosis, VaR, win rate, best/worst days
    """
    return {
        "mean"          : returns.mean(),
        "std"           : returns.std(),
        "skewness"      : returns.skew(),
        # pandas kurtosis() returns EXCESS kurtosis (normal distribution = 0)
        # positive = fatter tails than normal, negative = thinner tails
        "kurtosis"      : returns.kurtosis(),
        # VaR: on the worst 5% of days, losses exceed this threshold
        "var_95"        : returns.quantile(0.05),
        "best_day"      : returns.max(),
        "worst_day"     : returns.min(),
        "positive_days" : int((returns > 0).sum()),
        "negative_days" : int((returns < 0).sum()),
        "win_rate"      : (returns > 0).mean(),
    }


def print_distribution_summary(ticker: str, returns: pd.Series) -> None:
    """
    Print a formatted return distribution summary to the terminal.

    Args:
        ticker: Stock ticker symbol
        returns: Series of daily returns
    """
    s = distribution_stats(returns)

    # Interpret skewness in plain English
    if s["skewness"] < -0.5:
        skew_label = "negatively skewed (bad days worse than good days are good)"
    elif s["skewness"] > 0.5:
        skew_label = "positively skewed (good days better than bad days are bad)"
    else:
        skew_label = "roughly symmetric"

    # Interpret kurtosis in plain English
    if s["kurtosis"] > 1:
        kurt_label = "fat tails (extreme events more frequent than normal)"
    elif s["kurtosis"] < -1:
        kurt_label = "thin tails (fewer extreme events than normal)"
    else:
        kurt_label = "near-normal tails"

    print(f"\n{'='*50}")
    print(f"  Return Distribution Summary — {ticker}")
    print(f"{'='*50}")
    print(f"  Mean Daily Return  : {s['mean']*100:.4f}%")
    print(f"  Std Dev            : {s['std']*100:.4f}%")
    print(f"  ---")
    print(f"  Skewness           : {s['skewness']:.4f}  ({skew_label})")
    print(f"  Kurtosis (excess)  : {s['kurtosis']:.4f}  ({kurt_label})")
    print(f"  ---")
    print(f"  VaR (95%)          : {s['var_95']*100:.2f}%")
    print(f"  (On 5% of days, loss exceeds this threshold)")
    print(f"  ---")
    print(f"  Best Day           : +{s['best_day']*100:.2f}%")
    print(f"  Worst Day          : {s['worst_day']*100:.2f}%")
    print(f"  Positive Days      : {s['positive_days']:,}")
    print(f"  Negative Days      : {s['negative_days']:,}")
    print(f"  Win Rate           : {s['win_rate']*100:.1f}%")
    print(f"{'='*50}\n")


# ── Sharpe Ratio & Risk Comparison ───────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Risk-adjusted return: how much return per unit of risk.

    Formula: (Mean Annual Return - Risk Free Rate) / Annualized Volatility

    We annualize mean return by multiplying daily mean by 252.
    We annualize volatility by multiplying daily std by √252.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0.0 for simplicity)

    Returns:
        Float — Sharpe ratio. Higher is better.
        <0 = losing money, 0.5-1.0 = acceptable, >1.0 = good, >2.0 = excellent
    """
    mean_annual  = returns.mean() * TRADING_DAYS
    annual_vol   = returns.std()  * np.sqrt(TRADING_DAYS)

    if annual_vol == 0:
        return 0.0

    return (mean_annual - risk_free_rate) / annual_vol


def risk_profile(ticker: str, returns: pd.Series) -> dict:
    """
    Compute all risk metrics for one stock in one dict.
    Used to build the multi-stock comparison table.

    Args:
        ticker: Stock ticker symbol
        returns: Series of daily returns

    Returns:
        Dict with all Phase 3-7 metrics for this stock
    """
    dd = calculate_drawdown(returns)
    return {
        "Ticker"          : ticker,
        "Ann. Return"     : returns.mean() * TRADING_DAYS,
        "Ann. Volatility" : annualized_volatility(returns),
        "Sharpe Ratio"    : sharpe_ratio(returns),
        "Max Drawdown"    : max_drawdown(returns),
        "VaR (95%)"       : returns.quantile(0.05),
        "Win Rate"        : (returns > 0).mean(),
        "Curr. Drawdown"  : dd.iloc[-1],
    }


def build_comparison_table(tickers_returns: dict) -> "pd.DataFrame":
    """
    Build a multi-stock risk comparison DataFrame.

    Args:
        tickers_returns: Dict of {ticker: pd.Series of daily returns}

    Returns:
        DataFrame with one row per stock, sorted by Sharpe Ratio descending
    """
    import pandas as pd
    rows = [risk_profile(ticker, returns)
            for ticker, returns in tickers_returns.items()]

    df = pd.DataFrame(rows).set_index("Ticker")
    df = df.sort_values("Sharpe Ratio", ascending=False)
    return df


def print_comparison_table(df: "pd.DataFrame") -> None:
    """
    Print the risk comparison table in a readable terminal format.

    Args:
        df: DataFrame from build_comparison_table()
    """
    print(f"\n{'='*75}")
    print(f"  Multi-Stock Risk Comparison Table  (sorted by Sharpe Ratio)")
    print(f"{'='*75}")
    print(f"  {'Ticker':<8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} {'Max DD':>9} {'VaR95%':>8} {'WinRate':>8}")
    print(f"  {'-'*68}")

    for ticker, row in df.iterrows():
        print(
            f"  {ticker:<8}"
            f"  {row['Ann. Return']*100:>7.2f}%"
            f"  {row['Ann. Volatility']*100:>7.2f}%"
            f"  {row['Sharpe Ratio']:>8.3f}"
            f"  {row['Max Drawdown']*100:>7.2f}%"
            f"  {row['VaR (95%)']*100:>6.2f}%"
            f"  {row['Win Rate']*100:>6.1f}%"
        )

    print(f"{'='*75}")
    print(f"  Sharpe > 1.0 = good  |  Max DD closer to 0% = safer  |  Higher win rate = more consistent\n")