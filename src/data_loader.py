import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical OHLCV price data for a given stock ticker.

    Args:
        ticker: Stock symbol e.g. "AAPL", "MSFT", "RELIANCE.NS"
        period: How far back to fetch data.
                Options: "1mo", "3mo", "6mo", "1y", "2y", "5y"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Indexed by Date.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df


def fetch_multiple_stocks(tickers: list, period: str = "1y") -> dict:
    """
    Download historical data for multiple tickers at once.

    Args:
        tickers: List of ticker symbols e.g. ["AAPL", "MSFT", "GOOGL"]
        period: Same as fetch_stock_data

    Returns:
        Dictionary where key = ticker, value = DataFrame
        e.g. {"AAPL": df_apple, "MSFT": df_msft}
    """
    data = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        data[ticker] = fetch_stock_data(ticker, period)
    return data


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to the data/ folder as a CSV file.

    Args:
        df: The DataFrame to save
        filename: Name without extension e.g. "AAPL_1y"
    """
    path = f"data/{filename}.csv"
    df.to_csv(path)
    print(f"Saved to {path}")

def fetch_company_info(ticker: str) -> dict:
    """
    Fetch fundamental company data from Yahoo Finance.

    Retrieves business description, sector, size, and key
    valuation metrics for any publicly listed company.

    Args:
        ticker: Stock symbol e.g. "AAPL", "RELIANCE.NS"

    Returns:
        Dict of fundamental metrics. Missing values return "N/A".
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    def get(key, default="N/A"):
        val = info.get(key, default)
        return val if val is not None else default

    return {
        "name"          : get("longName"),
        "ticker"        : ticker,
        "sector"        : get("sector"),
        "industry"      : get("industry"),
        "country"       : get("country"),
        "employees"     : get("fullTimeEmployees"),
        "market_cap"    : get("marketCap"),
        "pe_ratio"      : get("trailingPE"),
        "eps"           : get("trailingEps"),
        "dividend_yield": get("dividendYield"),
        "beta"          : get("beta"),
        "52w_high"      : get("fiftyTwoWeekHigh"),
        "52w_low"       : get("fiftyTwoWeekLow"),
        "current_price" : get("currentPrice"),
        "currency"      : get("currency"),
    }