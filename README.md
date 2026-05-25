# AlgoTrader v2 — Production-Grade Algorithmic Trading Backtesting Framework

## Purpose

AlgoTrader v2 is a **production-grade algorithmic trading backtesting and analysis platform** designed for quantitative traders and financial analysts. It enables users to:

- **Backtest multiple quantitative trading strategies** on real NSE (National Stock Exchange) data
- **Evaluate strategy performance** using 15+ industry-standard metrics (Sharpe ratio, Sortino ratio, maximum drawdown, etc.)
- **Conduct out-of-sample testing** via walk-forward backtesting to avoid lookahead bias
- **Analyze market regimes** with automatic bull/bear market filtering
- **Perform parameter sensitivity analysis** to optimize strategy parameters
- **Run comprehensive unit tests** covering statistical methods (ADF, OLS, RSI)

The framework supports three distinct quantitative trading strategies: **pairs trading** (statistical arbitrage), **momentum** (trend-following), and **mean reversion** (oscillator-based). All backtests use real historical NSE Nifty 100 data with automatic dividend and split adjustment.

---

## Abstract

AlgoTrader v2 is an interactive Streamlit web application that implements a walk-forward backtesting engine with three proprietary trading strategies optimized for Indian equity markets. The system incorporates:

1. **Three Trading Strategies:**
   - **Pairs Trading Strategy** (Engle-Granger Cointegration): Identifies statistically cointegrated stock pairs and trades spread mean reversion using z-score signals
   - **Momentum Strategy** (Jegadeesh-Titman 1993): Cross-sectional momentum factor that longs top 20% performers and shorts bottom 20% over 3-month lookback, rebalancing monthly
   - **Mean Reversion Strategy** (Wilder RSI-14): Technical indicator-based strategy that goes long on RSI < 30 (oversold) and short on RSI > 70 (overbought)

2. **Advanced Backtesting Methodology:**
   - Walk-forward testing: calibrate on first 252 trading days, trade out-of-sample only (eliminates lookahead bias)
   - Market regime filter: suppresses short signals when Nifty 50 is in bull market (price > 200-day MA)
   - Transaction cost modeling (0.1% per trade)
   - 6.5% risk-free rate for Sharpe/Sortino calculations
   - 252 trading days per year assumption

3. **Real Market Data:**
   - Live NSE Nifty 100 constituent data via yfinance
   - Automatic dividend and split adjustment
   - Disk caching for fast subsequent loads
   - Synthetic fallback data for offline testing

4. **Comprehensive Analysis Tools:**
   - 15+ performance metrics (total return, Sharpe, Sortino, max drawdown, win rate, profit factor, etc.)
   - Equity curve visualization with benchmark comparison
   - Monthly/yearly returns heatmaps
   - Drawdown analysis
   - Parameter sensitivity charts (Sharpe ratio vs parameter sweep)
   - Trade log with entry/exit prices and P&L

5. **Quality Assurance:**
   - 28 unit tests covering statistical algorithms (ADF test, OLS regression, RSI calculation)
   - Backtester correctness validation
   - No lookahead bias verification

---

## Steps to Use

### 1. **Installation & Setup**

```bash
# Clone or navigate to the project directory
cd streamlit_algotrader

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- See `requirements.txt` for package versions

### 2. **Run Locally**

```bash
streamlit run app.py
```

This launches the web interface on `http://localhost:8501`

### 3. **Select Strategy & Parameters**

The app provides a sidebar interface to:
- **Choose Strategy:** Select from Pairs Trading, Momentum, or Mean Reversion
- **Adjust Strategy Parameters:** Each strategy has tunable parameters:
  - **Pairs Trading:** Min correlation, z-score entry/exit thresholds, formation window
  - **Momentum:** Lookback days, ranking quantile, rebalance frequency
  - **Mean Reversion:** RSI period, oversold/overbought thresholds
- **Set Backtesting Options:**
  - Enable/disable walk-forward testing
  - Enable/disable market regime filter
  - Adjust transaction costs
  - Set formation period (for out-of-sample calibration)

### 4. **Run Backtest**

Click the **"Run Backtest"** button to execute the strategy on real NSE data.

The backtester will:
1. Fetch Nifty 100 historical prices (with caching)
2. Generate trading signals using the selected strategy
3. Calculate daily portfolio returns accounting for transaction costs
4. Apply regime filter if enabled
5. Compute all 15+ performance metrics
6. Generate visualizations and trade logs

### 5. **Analyze Results**

The app displays 10 interactive tabs:

| Tab | Contents |
|-----|----------|
| **Strategy Info** | Strategy description, algorithm details, academic references, current parameters |
| **Backtest Results** | Key performance metrics (Sharpe, Sortino, max drawdown, etc.) |
| **Equity Curve** | Time-series plot of portfolio value vs Nifty 50 benchmark |
| **Monthly Returns** | Heatmap of monthly returns by year and month |
| **Yearly Returns** | Bar chart of annual returns |
| **Drawdown** | Maximum drawdown over time, underwater plot |
| **Trade Log** | Detailed table of all trades (entry/exit prices, P&L) |
| **Parameter Sensitivity** | 2D surface plot showing Sharpe ratio vs strategy parameters |
| **Run Tests** | Execute 28 unit tests covering statistical methods |
| **Data Source** | Shows where data is loaded from (yfinance vs synthetic fallback) |

### 6. **Run Unit Tests** (Optional)

Test the strategies and backtester offline:

```bash
python tests/test_strategies.py
```

This runs 28 unit tests covering:
- ADF (Augmented Dickey-Fuller) test correctness
- OLS regression accuracy
- RSI calculation bounds
- Backtester P&L validation
- No lookahead bias verification

---

## Outputs

### A. **Performance Metrics** (15+ metrics per strategy)

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative return from start to end of backtest period |
| **Annual Return** | Annualized return (%)  |
| **Sharpe Ratio** | Return per unit of risk (excess return / volatility), 6.5% RF rate |
| **Sortino Ratio** | Return per unit of downside risk (excess return / downside volatility) |
| **Maximum Drawdown** | Largest peak-to-trough decline (%) |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss ratio |
| **Average Win** | Mean return per winning trade (%) |
| **Average Loss** | Mean return per losing trade (%) |
| **Payoff Ratio** | Average win / Average loss |
| **Volatility** | Standard deviation of daily returns (annualized) |
| **Calmar Ratio** | Annual return / Max drawdown |
| **CAGR** | Compound annual growth rate |
| **Best Day** | Best daily return (%) |
| **Worst Day** | Worst daily return (%) |
| **Benchmark Return** | Nifty 50 total return over same period |
| **Beta** | Portfolio sensitivity to benchmark |

**Output Format:** Summary table displayed in "Backtest Results" tab with numeric precision and color-coding (green = outperformance, red = underperformance).

### B. **Visualizations**

1. **Equity Curve** (Line Chart)
   - X-axis: Date
   - Y-axis: Portfolio value (log scale)
   - Portfolio line vs Nifty 50 benchmark
   - Shaded regions for drawdown periods

2. **Monthly Returns Heatmap** (2D Grid)
   - Rows: Years
   - Columns: Months (Jan–Dec)
   - Color intensity: Return magnitude (green = positive, red = negative)
   - Values shown in each cell (%)

3. **Yearly Returns Bar Chart** (Bar Chart)
   - X-axis: Year
   - Y-axis: Annual return (%)
   - Green bars: Positive years
   - Red bars: Negative years

4. **Drawdown Analysis** (Area Chart)
   - X-axis: Date
   - Y-axis: Drawdown from peak (%)
   - Red shaded area below zero line

5. **Parameter Sensitivity Surface** (3D Surface/Contour)
   - X-axis: Parameter 1 (varies by strategy)
   - Y-axis: Parameter 2 (varies by strategy)
   - Z-axis: Sharpe ratio
   - Shows optimal parameter region visually

### C. **Trade Log** (Detailed Table)

Exported as interactive DataFrame with columns:

| Column | Description |
|--------|-------------|
| **Date** | Trade entry date |
| **Ticker** | Stock symbol |
| **Side** | LONG or SHORT |
| **Entry Price** | Entry price (₹) |
| **Exit Price** | Exit price (₹) |
| **Qty** | Number of shares |
| **P&L (₹)** | Profit/loss in rupees |
| **P&L (%)** | Profit/loss percentage |
| **Days Held** | Number of days position held |
| **Exit Date** | Trade exit date |

### D. **Strategy Information Report**

Each strategy tab displays:
- **Algorithm description** in plain English
- **Mathematical formulation** (equations)
- **Academic reference** (authors, year, journal)
- **Current parameters** with explanations
- **Interpretation guide** for results

### E. **Test Results** (Unit Tests Tab)

- **28 test cases** executed with pass/fail indicators
- Tests categories:
  - Statistical tests (ADF, OLS accuracy)
  - Indicator tests (RSI bounds, momentum calculation)
  - Backtester tests (P&L correctness, return matching)
  - Lookahead bias tests
- **Output:** ✓ Pass or ✗ Fail with error messages

### F. **Data Source Indicator** (Data Source Tab)

Shows:
- Data origin (yfinance live vs synthetic fallback)
- Download date & cache status
- Nifty 100 constituent list (date captured)
- Missing ticker handling
- Data quality checks (gaps, duplicates)

---

## Project Structure

```
streamlit_algotrader/
├── app.py                      ← Streamlit web app (entry point)
├── data_loader.py              ← yfinance data fetching + disk cache + synthetic fallback
├── base_strategy.py            ← Abstract Strategy base class
├── pairs_trading.py            ← Engle-Granger cointegration strategy
├── momentum.py                 ← Jegadeesh-Titman momentum strategy
├── mean_reversion.py           ← RSI mean reversion strategy
├── backtester.py               ← Walk-forward backtester + 15 metrics + regime filter
├── tests/
│   └── test_strategies.py      ← 28 unit tests (run: python tests/test_strategies.py)
├── requirements.txt            ← Python dependencies
├── .streamlit/config.toml      ← Dark theme configuration
└── README.md                   ← This file
```

---

## What's New in v2

- ✅ **Real NSE data** via yfinance (auto-adjusts splits & dividends, disk-cached)
- ✅ **Walk-forward backtesting** — calibrate on first 252 days, trade out-of-sample only
- ✅ **Market regime filter** — suppresses short signals when Nifty 50 > 200-day MA
- ✅ **28 unit tests** — ADF, OLS, RSI bounds, backtester correctness, no lookahead
- ✅ **Parameter sensitivity analysis** — Sharpe vs parameter sweep chart
- ✅ **10-tab UI** — includes tests tab and sensitivity tab
- ✅ **Production-grade backtester** — fully numpy-safe, no index-alignment errors

---

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud (free)
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file = `app.py`
4. Click Deploy

### Run Tests
```bash
python tests/test_strategies.py
```

---

## Key Features

- 🎯 **3 Quantitative Strategies** — Pairs Trading, Momentum, Mean Reversion
- 📊 **15+ Performance Metrics** — Sharpe, Sortino, Calmar, drawdown, etc.
- 🔄 **Walk-Forward Testing** — True out-of-sample validation, no lookahead bias
- 🛡️ **Regime Filter** — Suppress shorts in bull markets
- 📈 **Real NSE Data** — Live Nifty 100 constituent prices
- 🎨 **Interactive Dashboards** — 10 tabs with visualizations
- ✅ **28 Unit Tests** — Comprehensive test coverage
- 💾 **Disk Caching** — Fast data loading on subsequent runs

---

## Requirements

- Python 3.8+
- streamlit >= 1.32.0
- yfinance >= 0.2.36
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0

Install all with:
```bash
pip install -r requirements.txt
```

---

## Notes

- All backtests use **real NSE Nifty 100 data** with automatic adjustments for dividends and splits
- Strategies are **walk-forward tested** to avoid lookahead bias: calibration on first 252 trading days, out-of-sample trading thereafter
- **Market regime filter** automatically disables shorting when Nifty 50 is above its 200-day moving average
- **Transaction costs** modeled at 0.1% per round-trip trade
- **Risk-free rate** set to 6.5% (Indian government securities approximation) for Sharpe/Sortino calculations
- **252 trading days** assumed per calendar year

---

## License & References

Academic references for each strategy:
- **Momentum:** Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
- **Mean Reversion:** Wilder (1978) "New Concepts in Technical Trading Systems"
- **Pairs Trading:** Engle & Granger (1987) "Co-Integration and Error Correction"
- **Walk-Forward Testing:** Prado (2018) "Advances in Financial Machine Learning"
