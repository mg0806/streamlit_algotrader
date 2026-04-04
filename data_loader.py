"""
data_loader.py — Production-ready DataLoader

Primary:  Downloads real NSE data via yfinance (^NSEI benchmark + .NS tickers)
Fallback: Generates statistically realistic synthetic data if yfinance unavailable

Handles:
  - Delisted / unavailable tickers (dropped gracefully)
  - Missing data / trading holidays (forward-filled)
  - Corporate actions (splits & dividends) via auto_adjust=True
  - Stocks with >20% missing data (removed from universe)
  - Disk caching so repeated runs don't re-download
"""

import os, pickle, hashlib, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

NIFTY100_TICKERS = [
    "HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN",
    "INDUSINDBK","FEDERALBNK","BANDHANBNK","IDFCFIRSTB","AUBANK",
    "TCS","INFY","WIPRO","HCLTECH","TECHM",
    "LTIM","PERSISTENT","COFORGE","MPHASIS","OFSS",
    "RELIANCE","ONGC","IOC","BPCL","GAIL",
    "PETRONET","MGL","IGL","POWERGRID","NTPC",
    "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR",
    "MARICO","GODREJCP","COLPAL","EMAMILTD","TATACONSUM",
    "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO",
    "EICHERMOT","TVSMOTOR","ASHOKLEY","BALKRISIND","APOLLOTYRE",
    "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","TORNTPHARM",
    "BIOCON","LUPIN","AUROPHARMA","IPCALAB","ALKEM",
    "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","COALINDIA",
    "NMDC","SAIL","NATIONALUM","HINDCOPPER","MOIL",
    "LT","ADANIENT","ADANIPORTS","ULTRACEMCO","SHREECEM",
    "AMBUJACEM","ACC","DALBHARAT","JKCEMENT","NUVOCO",
    "BAJFINANCE","BAJAJFINSV","HDFCLIFE","SBILIFE","ICICIGI",
    "MUTHOOTFIN","CHOLAFIN","LICHSGFIN","SHRIRAMFIN","ABCAPITAL",
    "TITAN","ASIANPAINT","PIDILITIND","HAVELLS","VOLTAS",
    "WHIRLPOOL","CROMPTON","DIXON","VGUARD","ORIENTELEC",
]

SECTOR_MAP = {
    "HDFCBANK":"Banking","ICICIBANK":"Banking","KOTAKBANK":"Banking","AXISBANK":"Banking","SBIN":"Banking",
    "INDUSINDBK":"Banking","FEDERALBNK":"Banking","BANDHANBNK":"Banking","IDFCFIRSTB":"Banking","AUBANK":"Banking",
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","PERSISTENT":"IT","COFORGE":"IT","MPHASIS":"IT","OFSS":"IT",
    "RELIANCE":"Energy","ONGC":"Energy","IOC":"Energy","BPCL":"Energy","GAIL":"Energy",
    "PETRONET":"Energy","MGL":"Energy","IGL":"Energy","POWERGRID":"Energy","NTPC":"Energy",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG","DABUR":"FMCG",
    "MARICO":"FMCG","GODREJCP":"FMCG","COLPAL":"FMCG","EMAMILTD":"FMCG","TATACONSUM":"FMCG",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto","HEROMOTOCO":"Auto",
    "EICHERMOT":"Auto","TVSMOTOR":"Auto","ASHOKLEY":"Auto","BALKRISIND":"Auto","APOLLOTYRE":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma","TORNTPHARM":"Pharma",
    "BIOCON":"Pharma","LUPIN":"Pharma","AUROPHARMA":"Pharma","IPCALAB":"Pharma","ALKEM":"Pharma",
    "TATASTEEL":"Metals","JSWSTEEL":"Metals","HINDALCO":"Metals","VEDL":"Metals","COALINDIA":"Metals",
    "NMDC":"Metals","SAIL":"Metals","NATIONALUM":"Metals","HINDCOPPER":"Metals","MOIL":"Metals",
    "LT":"Infra","ADANIENT":"Infra","ADANIPORTS":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEM":"Infra","ACC":"Infra","DALBHARAT":"Infra","JKCEMENT":"Infra","NUVOCO":"Infra",
    "BAJFINANCE":"FinSvc","BAJAJFINSV":"FinSvc","HDFCLIFE":"FinSvc","SBILIFE":"FinSvc","ICICIGI":"FinSvc",
    "MUTHOOTFIN":"FinSvc","CHOLAFIN":"FinSvc","LICHSGFIN":"FinSvc","SHRIRAMFIN":"FinSvc","ABCAPITAL":"FinSvc",
    "TITAN":"ConsDisc","ASIANPAINT":"ConsDisc","PIDILITIND":"ConsDisc","HAVELLS":"ConsDisc","VOLTAS":"ConsDisc",
    "WHIRLPOOL":"ConsDisc","CROMPTON":"ConsDisc","DIXON":"ConsDisc","VGUARD":"ConsDisc","ORIENTELEC":"ConsDisc",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")


class DataLoader:
    """
    Loads historical OHLCV data for the Nifty 100 universe.
    Priority: disk cache → yfinance live → synthetic fallback
    """

    def __init__(self, start_date="2020-01-01", end_date="2024-01-01",
                 universe=None, transaction_cost=0.001,
                 cache_days=1, use_cache=True):
        self.start_date       = pd.Timestamp(start_date)
        self.end_date         = pd.Timestamp(end_date)
        self.universe         = universe or NIFTY100_TICKERS
        self.transaction_cost = transaction_cost
        self.cache_days       = cache_days
        self.use_cache        = use_cache
        self.prices_df        = None
        self.returns_df       = None
        self.benchmark_returns= None
        self.data_source      = None
        os.makedirs(CACHE_DIR, exist_ok=True)

    def load(self) -> pd.DataFrame:
        cache_path = os.path.join(CACHE_DIR, f"{self._cache_key()}.pkl")

        # 1. Disk cache
        if self.use_cache and self._cache_valid(cache_path):
            print("Loading from disk cache...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.prices_df         = cached["prices"]
            self.returns_df        = cached["returns"]
            self.benchmark_returns = cached["benchmark"]
            self.data_source       = "cache"
            self._print_summary()
            return self.prices_df

        # 2. yfinance
        try:
            self._load_yfinance()
            self.data_source = "yfinance"
            with open(cache_path, "wb") as f:
                pickle.dump({"prices": self.prices_df,
                             "returns": self.returns_df,
                             "benchmark": self.benchmark_returns}, f)
            print("Data cached to disk.")
        except Exception as e:
            print(f"yfinance unavailable ({type(e).__name__}: {e}). Using synthetic data.")
            print("To use real data: pip install yfinance")
            self._load_synthetic()
            self.data_source = "synthetic"

        self._print_summary()
        return self.prices_df

    def _load_yfinance(self):
        import yfinance as yf
        print(f"Downloading {len(self.universe)} NSE stocks from Yahoo Finance...")
        yf_tickers = [t + ".NS" for t in self.universe]

        raw = yf.download(
            tickers=yf_tickers, start=self.start_date, end=self.end_date,
            auto_adjust=True, progress=False, threads=True,
        )
        if raw.empty:
            raise ValueError("yfinance returned empty DataFrame")

        prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
        prices.columns = [str(c).replace(".NS", "") for c in prices.columns]
        valid  = [t for t in self.universe if t in prices.columns]
        prices = prices[valid]

        # Drop stocks with >20% missing
        prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.80))
        prices = prices.ffill().bfill().dropna(how="all")

        if len(prices.columns) < 10:
            raise ValueError(f"Too few stocks loaded: {len(prices.columns)}")

        self.prices_df  = prices
        self.returns_df = prices.pct_change().dropna()

        print("Downloading Nifty 50 benchmark (^NSEI)...")
        nifty = yf.download("^NSEI", start=self.start_date, end=self.end_date,
                            auto_adjust=True, progress=False)
        if nifty.empty:
            raise ValueError("Could not download Nifty 50 benchmark")
        # squeeze() converts single-column DataFrame → Series on all yfinance versions
        bm = nifty["Close"].squeeze().pct_change().dropna()
        bm = bm.reindex(self.returns_df.index).ffill().fillna(0)
        bm = pd.Series(bm.values.ravel().astype(float),   # always 1-D
                       index=bm.index, name="NIFTY50")
        self.benchmark_returns = bm

    def _load_synthetic(self):
        np.random.seed(42)
        days = pd.bdate_range(self.start_date, self.end_date, freq="B")
        T, N = len(days), len(self.universe)

        # GARCH(1,1) market
        ω, α, β_g = 2e-6, 0.09, 0.88
        mkt = np.zeros(T)
        h   = ω / (1 - α - β_g)
        for t in range(T):
            h       = ω + α*(mkt[t-1]**2 if t > 0 else 0) + β_g*h
            mkt[t]  = 0.0004 + np.sqrt(h) * np.random.standard_t(5)/np.sqrt(5/3)

        sectors  = list(set(SECTOR_MAP.get(t, "Other") for t in self.universe))
        sec_r    = {s: np.random.normal(0, 0.009, T) for s in sectors}
        all_r    = np.zeros((T, N))

        for j, ticker in enumerate(self.universe):
            s      = SECTOR_MAP.get(ticker, "Other")
            dv     = np.random.uniform(0.18, 0.38) / np.sqrt(252)
            bm_    = np.random.uniform(0.6, 1.4)
            bs_    = np.random.uniform(0.3, 0.7)
            iv     = dv * np.sqrt(max(0.1, 1 - bm_**2*0.04 - bs_**2*0.01))
            all_r[:, j] = (np.random.uniform(-0.0001, 0.0006)
                           + bm_*mkt + bs_*sec_r[s]
                           + np.random.normal(0, iv, T))

        # Inject cointegration
        seen = {}
        for i, ticker in enumerate(self.universe):
            s = SECTOR_MAP.get(ticker, "Other")
            if s not in seen:
                seen[s] = i
            else:
                j, hr, kp = seen[s], np.random.uniform(0.7,1.3), np.random.uniform(0.05,0.15)
                sp = np.zeros(T)
                for t in range(1, T):
                    sp[t] = sp[t-1] - kp*sp[t-1] + np.random.normal(0, 0.005)
                all_r[:, i] = hr*all_r[:, j] + np.diff(np.concatenate([[0], sp]))
                seen[s] = i

        prices = np.exp(np.log(np.random.uniform(100, 3000, N)) + np.cumsum(all_r, axis=0))
        df = pd.DataFrame(prices.round(2), index=days, columns=self.universe)
        df.index.name = "Date"
        self.prices_df  = df
        self.returns_df = df.pct_change().dropna()
        bm = pd.Series(mkt, index=days, name="NIFTY50")
        self.benchmark_returns = bm.reindex(self.returns_df.index).fillna(0)

    def _cache_key(self):
        return hashlib.md5(f"{self.start_date}_{self.end_date}_{len(self.universe)}".encode()).hexdigest()[:12]

    def _cache_valid(self, path):
        if not os.path.exists(path): return False
        return (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))) < timedelta(days=self.cache_days)

    def _print_summary(self):
        print(f"  Source: {self.data_source.upper()} | "
              f"Stocks: {len(self.prices_df.columns)} | "
              f"Days: {len(self.prices_df)} | "
              f"{str(self.start_date)[:10]} to {str(self.end_date)[:10]}")

    def get_prices(self)            -> pd.DataFrame: return self.prices_df
    def get_returns(self)           -> pd.DataFrame: return self.returns_df
    def get_benchmark_returns(self) -> pd.Series:    return self.benchmark_returns
    def get_universe(self)          -> list:         return list(self.prices_df.columns)
    def get_data_source(self)       -> str:          return self.data_source
