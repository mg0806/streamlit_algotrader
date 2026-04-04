# AlgoTrader v2 — Nifty 100 Backtesting Framework

## What's New in v2
- **Real NSE data** via yfinance (auto-adjusts splits & dividends, disk-cached)
- **Walk-forward backtesting** — calibrate on first 252 days, trade out-of-sample only
- **Market regime filter** — suppresses short signals when Nifty 50 > 200-day MA
- **28 unit tests** — ADF, OLS, RSI bounds, backtester correctness, no lookahead
- **Parameter sensitivity analysis** — Sharpe vs parameter sweep chart
- **10-tab UI** — includes tests tab and sensitivity tab

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (free)
1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app
3. Select repo, set main file = `app.py`
4. Click Deploy

## Project Structure
```
├── app.py                  ← Streamlit app (entry point)
├── data_loader.py          ← yfinance + synthetic fallback + disk cache
├── base_strategy.py        ← Abstract Strategy base class
├── pairs_trading.py        ← Engle-Granger cointegration
├── momentum.py             ← Jegadeesh-Titman 1993
├── mean_reversion.py       ← Wilder RSI-14
├── backtester.py           ← Walk-forward + regime filter + 15 metrics
├── tests/
│   └── test_strategies.py  ← 28 unit tests (run: python tests/test_strategies.py)
├── requirements.txt
└── .streamlit/config.toml  ← Dark theme
```
