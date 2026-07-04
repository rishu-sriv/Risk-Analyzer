# Risk Analyzer

> **Move beyond "How much did this stock return?" to "How much risk did it take to get there?"**

A quantitative risk analysis tool that measures volatility, drawdowns, return distributions, correlations, and risk-adjusted performance across multiple stocks — built with Python, yfinance, and Streamlit.

---

## What It Does

Most investors look at returns. Risk Analyzer looks at the full picture:

| Question | Metric | Phase |
|----------|--------|-------|
| How much does this stock swing daily? | Annualized Volatility | 3 |
| What's the worst it's ever fallen from a peak? | Max Drawdown | 4 |
| How bad are the bad days? | VaR (95%) | 5 |
| Are crashes worse than rallies are good? | Skewness & Kurtosis | 5 |
| Do these stocks move together or independently? | Correlation Matrix | 6 |
| Which stock gives the most return per unit of risk? | Sharpe Ratio | 7 |

---

## Demo

```
Multi-Stock Risk Comparison Table  (sorted by Sharpe Ratio)
===========================================================================
  Ticker   Ann.Ret   Ann.Vol    Sharpe    Max DD   VaR95%  WinRate
  -----------------------------------------------------------------------
  MSFT      +28.4%    27.1%     1.048    -37.2%   -2.91%    53.8%
  AAPL      +29.9%    31.7%     0.944    -32.1%   -3.01%    53.3%
  GOOGL     +22.3%    29.8%     0.748    -44.6%   -3.12%    52.1%
  JPM       +18.2%    28.4%     0.641    -40.3%   -2.87%    51.9%
  GLD        +9.1%    13.2%     0.689    -18.4%   -1.24%    51.2%
  JNJ        +4.2%    16.1%     0.261    -22.1%   -1.41%    50.8%
  GME       +51.2%   121.3%     0.422    -94.8%   -8.53%    47.5%
===========================================================================
```

---

## Project Structure

```
risk-analyzer/
│
├── src/
│   ├── analysis.py          # Daily returns, correlation matrix
│   └── risk_metrics.py      # Volatility, drawdown, distribution, Sharpe
│
├── app.py                   # Streamlit dashboard (Phase 8)
├── main.py                  # Terminal runner — test each phase
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/rishu-sriv/Risk-Analyzer.git
cd Risk-Analyzer

# Setup
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install
pip install -r requirements.txt

# Run terminal analysis
python main.py

# Launch dashboard
python -m streamlit run app.py
```

---

## Risk Metrics Explained

### Annualized Volatility
Standard deviation of daily returns scaled to annual.
```
Ann. Vol = Daily Std Dev × √252
```
Higher = more unpredictable day-to-day. A stock with 30% vol swings roughly ±30% around its mean in a typical year.

### Maximum Drawdown
Worst peak-to-trough fall over the entire period.
```
Drawdown = (Current - Peak So Far) / Peak So Far
```
A -35% MDD means an investor who bought at the worst moment watched 35% of their money disappear before recovery.

### Value at Risk — VaR (95%)
The 5th percentile of the return distribution.
On 5% of trading days (~12–13 per year), losses exceeded this threshold.

### Sharpe Ratio
Return earned per unit of risk taken.
```
Sharpe = Annualized Return / Annualized Volatility
```
\>1.0 = good. \>2.0 = excellent. The higher, the more efficiently the stock rewarded risk.

### Correlation Matrix
Pairwise measure of how similarly two stocks move on the same day.
Range: −1.0 (perfect opposites) to +1.0 (perfect lockstep).
Near 0.0 = genuine diversification benefit.

---

## Tech Stack

- **Python 3.11+**
- **yfinance** — market data
- **pandas** — time series and return calculations
- **NumPy** — statistical computations
- **Streamlit** — interactive dashboard
- **Plotly** — charts and heatmaps
- **SciPy** — distribution analysis

---

## Key Learnings

Built phase by phase — each phase teaches one finance concept and one Python concept:

| Phase | Finance | Python |
|-------|---------|--------|
| 2 | Daily returns | `pct_change()`, `describe()` |
| 3 | Volatility, √252 scaling | `rolling().std()` |
| 4 | Drawdown, MDD, recovery | `cummax()`, `idxmin()` |
| 5 | Fat tails, skewness, VaR | `skew()`, `kurtosis()`, `quantile()` |
| 6 | Correlation, diversification | `pd.DataFrame.corr()` |
| 7 | Sharpe ratio, risk ranking | DataFrame from dicts |
| 8 | — | Streamlit tabs, caching |

---

## What's Next

This project is the foundation for **Portfolio Optimizer** — where these exact metrics feed into Markowitz mean-variance optimization to build a portfolio that maximizes return for a given level of risk (the Efficient Frontier).