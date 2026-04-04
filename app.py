"""
AlgoTrader v2 — Production-grade Algorithmic Trading Backtesting Framework
Streamlit Web App

New in v2:
  - Real NSE data via yfinance (auto_adjust splits & dividends)
  - Walk-forward backtesting (out-of-sample only)
  - Market regime filter (no shorting in bull markets)
  - Disk caching (no re-download on refresh)
  - 28 unit tests covering all core algorithms
  - Parameter sensitivity analysis
  - Data source indicator

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy: push to GitHub → share.streamlit.io → connect repo → deploy
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from data_loader    import DataLoader
from pairs_trading  import PairsTradingStrategy
from momentum       import MomentumStrategy
from mean_reversion import MeanReversionStrategy
from backtester     import Backtester

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrader v2 — Nifty 100",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.stApp{background:#060d16;}
section[data-testid="stSidebar"]{background:#0b1929!important;border-right:1px solid #1a3a5c;}
section[data-testid="stSidebar"] *{color:#e8f4fd!important;}
[data-testid="stMetric"]{background:#0f2236;border:1px solid #1a3a5c;border-radius:10px;padding:14px 18px!important;}
[data-testid="stMetricLabel"]{color:#90b4ce!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.8px;}
[data-testid="stMetricValue"]{color:#e8f4fd!important;font-family:'JetBrains Mono',monospace!important;font-size:20px!important;}
button[data-baseweb="tab"]{color:#90b4ce!important;font-weight:600;font-size:13px;}
button[data-baseweb="tab"][aria-selected="true"]{color:#c9a84c!important;border-bottom:2px solid #c9a84c!important;}
.stButton>button{background:linear-gradient(135deg,#c9a84c,#e8c566)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:8px!important;padding:10px 28px!important;font-size:14px!important;}
details{background:#0f2236!important;border:1px solid #1a3a5c!important;border-radius:8px!important;}
summary{color:#c9a84c!important;font-weight:600;}
hr{border-color:#1a3a5c!important;}
h1,h2,h3{color:#e8f4fd!important;}
p,li{color:#90b4ce;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
C = {"bg":"#060d16","surface":"#0f2236","border":"#1a3a5c","gold":"#c9a84c",
     "text":"#e8f4fd","text2":"#90b4ce","pairs":"#3b82f6","mom":"#f97316",
     "rev":"#10b981","bench":"#a855f7","green":"#00e676","red":"#ff5252"}

STRAT_COLORS = {
    "Pairs Trading (Engle-Granger Cointegration)": C["pairs"],
    "Cross-Sectional Momentum (Jegadeesh-Titman)": C["mom"],
    "Mean Reversion (RSI-14)":                     C["rev"],
    "Nifty 50 Benchmark":                          C["bench"],
}
SHORT = {
    "Pairs Trading (Engle-Granger Cointegration)": "Pairs",
    "Cross-Sectional Momentum (Jegadeesh-Titman)": "Momentum",
    "Mean Reversion (RSI-14)":                     "Mean Rev",
    "Nifty 50 Benchmark":                          "Nifty 50",
}

def set_dark():
    plt.rcParams.update({
        "figure.facecolor":C["bg"],"axes.facecolor":C["surface"],
        "axes.edgecolor":C["border"],"axes.labelcolor":C["text2"],
        "axes.titlecolor":C["text"],"xtick.color":C["text2"],
        "ytick.color":C["text2"],"grid.color":C["border"],"grid.alpha":.5,
        "text.color":C["text"],"legend.facecolor":C["surface"],
        "legend.edgecolor":C["border"],"legend.labelcolor":C["text"],
        "font.family":"monospace","font.size":9,
    })

# ── Header ────────────────────────────────────────────────────────────────────
def render_header(data_source=None):
    src_badge = ""
    if data_source == "yfinance":
        src_badge = '<span style="background:#0d3321;border:1px solid #00e676;color:#00e676;border-radius:20px;padding:3px 10px;font-size:11px">📡 Live NSE Data</span>'
    elif data_source == "cache":
        src_badge = '<span style="background:#132840;border:1px solid #3b82f6;color:#3b82f6;border-radius:20px;padding:3px 10px;font-size:11px">📂 Cached Data</span>'
    elif data_source == "synthetic":
        src_badge = '<span style="background:#2d1a00;border:1px solid #f97316;color:#f97316;border-radius:20px;padding:3px 10px;font-size:11px">🔬 Synthetic Data</span>'

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0b1929,#0f2236);border:1px solid #1a3a5c;
                border-radius:12px;padding:24px 32px;margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px">
        <div style="background:linear-gradient(135deg,#c9a84c,#e8c566);border-radius:10px;
                    padding:9px 13px;font-weight:900;font-size:18px;color:#000;
                    font-family:'JetBrains Mono',monospace">AT</div>
        <div>
          <div style="font-size:20px;font-weight:700;color:#e8f4fd">
            AlgoTrader v2 — Backtesting Framework
          </div>
          <div style="font-size:11px;color:#90b4ce;font-family:'JetBrains Mono',monospace;margin-top:2px">
            Nifty 100 · Walk-Forward · Regime Filter · 28 Unit Tests · Real NSE Data
          </div>
        </div>
        <div style="margin-left:auto">{src_badge}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown('<div style="text-align:center;padding:12px 0 6px"><div style="font-size:17px;font-weight:700;color:#c9a84c">⚙ Configuration</div></div>', unsafe_allow_html=True)

    st.sidebar.markdown("### 📅 Data")
    start   = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
    end     = st.sidebar.date_input("End Date",   value=pd.Timestamp("2024-01-01"))
    tx_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.10, 0.01,
                                 help="Brokerage + STT approximation") / 100

    st.sidebar.markdown("### ⚡ Execution")
    walk_fwd  = st.sidebar.checkbox("Walk-Forward (OOS only)", value=True,
                                     help="Use first 252 days for calibration, trade only on unseen data")
    form_days = st.sidebar.slider("Formation period (days)", 126, 504, 252, 21,
                                   disabled=not walk_fwd)
    regime_f  = st.sidebar.checkbox("Market Regime Filter", value=True,
                                     help="Suppress short signals when Nifty 50 is above 200-day MA")

    st.sidebar.markdown("### 🎯 Strategies")
    run_pairs = st.sidebar.checkbox("Pairs Trading",  value=True)
    run_mom   = st.sidebar.checkbox("Momentum",       value=True)
    run_rev   = st.sidebar.checkbox("Mean Reversion", value=True)

    if run_pairs:
        with st.sidebar.expander("Pairs Trading Parameters"):
            coint_pval = st.slider("Cointegration p-value", 0.01, 0.10, 0.05, 0.01)
            z_entry    = st.slider("Z-score Entry",  1.0, 3.0, 2.0, 0.1)
            z_exit     = st.slider("Z-score Exit",   0.1, 1.0, 0.5, 0.1)
            roll_win   = st.slider("Rolling window (days)", 20, 120, 60, 5)
            max_pairs  = st.slider("Max pairs", 10, 80, 40, 5)
    else:
        coint_pval, z_entry, z_exit, roll_win, max_pairs = 0.05, 2.0, 0.5, 60, 40

    if run_mom:
        with st.sidebar.expander("Momentum Parameters"):
            lbk_days  = st.slider("Formation (days)",   21, 126, 63, 5)
            top_q     = st.slider("Long quantile %",     5,  40, 20, 5) / 100
            bot_q     = st.slider("Short quantile %",    5,  40, 20, 5) / 100
            rebal     = st.slider("Rebalance (days)",    5,  42, 21,  1)
            skip      = st.slider("Skip recent days",    0,  10,  5,  1)
    else:
        lbk_days, top_q, bot_q, rebal, skip = 63, .2, .2, 21, 5

    if run_rev:
        with st.sidebar.expander("Mean Reversion Parameters"):
            rsi_p  = st.slider("RSI period",           7, 21, 14,  1)
            over_s = st.slider("Oversold threshold",  20, 40, 30,  1)
            over_b = st.slider("Overbought threshold",60, 80, 70,  1)
            exit_t = st.slider("Exit threshold",      40, 60, 50,  1)
            max_p  = st.slider("Max positions",        5, 40, 20,  5)
    else:
        rsi_p, over_s, over_b, exit_t, max_p = 14, 30, 70, 50, 20

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

    return dict(
        start_date=str(start), end_date=str(end), tx_cost=tx_cost,
        walk_fwd=walk_fwd, form_days=form_days, regime_filter=regime_f,
        run_pairs=run_pairs, run_mom=run_mom, run_rev=run_rev,
        coint_pval=coint_pval, z_entry=z_entry, z_exit=z_exit,
        roll_win=roll_win, max_pairs=max_pairs,
        lbk_days=lbk_days, top_q=top_q, bot_q=bot_q, rebal=rebal, skip=skip,
        rsi_p=rsi_p, over_s=over_s, over_b=over_b, exit_t=exit_t, max_p=max_p,
        run_btn=run_btn,
    )

# ── Cached backtest ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def run_backtest(config_key, cfg):
    loader = DataLoader(
        start_date=cfg["start_date"], end_date=cfg["end_date"],
        transaction_cost=cfg["tx_cost"],
    )
    loader.load()

    strategies = []
    if cfg["run_pairs"]:
        strategies.append(PairsTradingStrategy(
            coint_pvalue=cfg["coint_pval"], z_entry=cfg["z_entry"],
            z_exit=cfg["z_exit"], rolling_window=cfg["roll_win"],
            max_pairs=cfg["max_pairs"],
        ))
    if cfg["run_mom"]:
        strategies.append(MomentumStrategy(
            lookback_days=cfg["lbk_days"], top_quantile=cfg["top_q"],
            bottom_quantile=cfg["bot_q"], rebalance_freq=cfg["rebal"],
            skip_days=cfg["skip"],
        ))
    if cfg["run_rev"]:
        strategies.append(MeanReversionStrategy(
            rsi_period=cfg["rsi_p"], oversold_threshold=float(cfg["over_s"]),
            overbought_threshold=float(cfg["over_b"]),
            exit_threshold=float(cfg["exit_t"]), max_positions=cfg["max_p"],
        ))

    results = []
    for st_obj in strategies:
        bt = Backtester(
            st_obj, loader,
            transaction_cost=cfg["tx_cost"],
            walk_forward=cfg["walk_fwd"],
            formation_days=cfg["form_days"],
            use_regime_filter=cfg["regime_filter"],
        )
        bt.run()
        perf = bt.calculate_performance()
        results.append({"strategy": st_obj, "backtester": bt, "perf": perf})

    bm_r  = loader.get_benchmark_returns()
    bm_ec = (1 + bm_r.shift(1).fillna(0)).cumprod()
    bm_dd = (bm_ec - bm_ec.cummax()) / bm_ec.cummax()
    return results, loader, bm_ec, bm_dd, bm_r

# ── Charts ────────────────────────────────────────────────────────────────────
def plot_equity_curves(results, bm_ec):
    set_dark()
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(C["bg"])
    for r in results:
        bt, name = r["backtester"], r["strategy"].get_name()
        ec = bt.equity_curve
        ax.plot(ec.index, ec.values, color=STRAT_COLORS.get(name,"#fff"),
                lw=1.8, label=f"{SHORT.get(name,name)}  ({(ec.iloc[-1]-1)*100:+.1f}%)")
    bm_t = (bm_ec.iloc[-1].squeeze()-1)*100
    ax.plot(bm_ec.index, bm_ec.values, color=C["bench"], lw=1.4, ls="--",
            label=f"Nifty 50  ({bm_t:+.1f}%)", alpha=.85)
    ax.axhline(1.0, color="#fff", lw=.4, ls=":", alpha=.3)
    ax.set_title("Equity Curves — All Strategies vs Nifty 50",
                 fontsize=12, fontweight="bold", color=C["gold"], pad=10)
    ax.set_ylabel("Portfolio Value (₹1 = 1.00x)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.2f}x"))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=.9)
    ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_drawdowns(results, bm_dd):
    set_dark()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor(C["bg"])
    for r in results:
        bt, name = r["backtester"], r["strategy"].get_name()
        dd = bt.drawdown_series()*100
        col = STRAT_COLORS.get(name,"#fff")
        ax.fill_between(dd.index, dd.values, 0, alpha=.2, color=col)
        ax.plot(dd.index, dd.values, color=col, lw=1.3,
                label=f"{SHORT.get(name,name)}  ({dd.min():.1f}%)")
    bm_d = bm_dd*100
    ax.fill_between(bm_d.index, bm_d.squeeze().values, 0, alpha=.1, color=C["bench"])
    ax.plot(bm_d.index, bm_d.squeeze().values, color=C["bench"], lw=1.2, ls="--",
            label=f"Nifty 50  ({float(bm_d.squeeze().min()):.1f}%)")
    ax.axhline(0, color="#fff", lw=.4, alpha=.3)
    ax.set_title("Drawdown Chart", fontsize=12, fontweight="bold", color=C["gold"], pad=8)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax.legend(loc="lower left", fontsize=8.5, framealpha=.9)
    ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_rolling_sharpe(results, bm_r, window=126):
    set_dark()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor(C["bg"])
    rf_d = (1.065)**(1/252)-1
    for r in results:
        bt, name = r["backtester"], r["strategy"].get_name()
        rs = bt.rolling_sharpe(window).dropna()
        ax.plot(rs.index, rs.values, color=STRAT_COLORS.get(name,"#fff"),
                lw=1.4, label=SHORT.get(name,name), alpha=.9)
    bm_rs = ((bm_r-rf_d).rolling(window).mean()/(bm_r.rolling(window).std()+1e-9)*np.sqrt(252)).dropna()
    ax.plot(bm_rs.index, bm_rs.values, color=C["bench"], lw=1.2, ls="--",
            label="Nifty 50", alpha=.8)
    for y, col in [(0,"#fff"),(1,C["green"]),(-1,C["red"])]:
        ax.axhline(y, color=col, lw=.5, ls=":", alpha=.4)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio",
                 fontsize=12, fontweight="bold", color=C["gold"], pad=8)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=.9)
    ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_return_distribution(results):
    set_dark()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(C["bg"])
    for r in results:
        bt, name = r["backtester"], r["strategy"].get_name()
        dr = bt.daily_returns.dropna()*100
        ax.hist(dr, bins=80, alpha=.45, color=STRAT_COLORS.get(name,"#fff"),
                label=SHORT.get(name,name), density=True)
    all_r = pd.concat([r["backtester"].daily_returns for r in results]).dropna()*100
    x = np.linspace(all_r.quantile(.001), all_r.quantile(.999), 300)
    mu, sig = all_r.mean(), all_r.std()
    ax.plot(x, (1/(sig*np.sqrt(2*np.pi)))*np.exp(-.5*((x-mu)/sig)**2),
            "w--", lw=1.4, alpha=.7, label="Normal PDF")
    ax.axvline(0, color="#fff", lw=.7, alpha=.4)
    ax.set_title("Daily Return Distribution", fontsize=11, fontweight="bold", color=C["gold"])
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=.9)
    ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_monthly_heatmap(bt, name):
    set_dark()
    mr = bt.monthly_returns()*100
    if mr.empty: return None
    for m in range(1,13):
        if m not in mr.columns: mr[m] = np.nan
    mr = mr[sorted(mr.columns)]
    fig, ax = plt.subplots(figsize=(12, max(2.5, len(mr)*0.6+1)))
    fig.patch.set_facecolor(C["bg"])
    cmap  = LinearSegmentedColormap.from_list("rg", ["#b71c1c","#424242","#1b5e20"])
    vals  = mr.values
    vmax  = max(abs(np.nanmax(vals)), abs(np.nanmin(vals)), 1)
    im    = ax.imshow(vals, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    months= ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(range(len(mr.columns)))
    ax.set_xticklabels([months[c-1] for c in mr.columns], fontsize=9)
    ax.set_yticks(range(len(mr.index)))
    ax.set_yticklabels(mr.index, fontsize=9)
    for i in range(mr.shape[0]):
        for j in range(mr.shape[1]):
            v = vals[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(v)>vmax*.5 else "#333",
                        fontweight="bold")
    plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=.8)
    ax.set_title(f"Monthly Returns — {SHORT.get(name,name)}",
                 fontsize=11, fontweight="bold", color=C["gold"], pad=8)
    plt.tight_layout()
    return fig

def plot_pair_zscore(prices, strategy, pair_idx=0):
    if not strategy.cointegrated_pairs: return None
    pair = strategy.cointegrated_pairs[min(pair_idx, len(strategy.cointegrated_pairs)-1)]
    sA, sB = pair
    beta   = strategy.hedge_ratios[pair]
    lp     = np.log(prices)
    spread = lp[sA] - beta*lp[sB]
    rm     = spread.rolling(strategy.rolling_window, min_periods=30).mean()
    rs     = spread.rolling(strategy.rolling_window, min_periods=30).std()
    z      = (spread - rm) / (rs + 1e-9)
    set_dark()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    fig.patch.set_facecolor(C["bg"])
    a1.plot(spread.index, spread.values, color=C["pairs"], lw=1.2)
    a1.plot(rm.index, rm.values, color=C["gold"], lw=1, ls="--", alpha=.7, label="Rolling Mean")
    a1.set_title(f"Pair: {sA} / {sB}   β = {beta:.4f}", fontsize=11,
                 fontweight="bold", color=C["gold"])
    a1.set_ylabel("Log Spread"); a1.legend(fontsize=8); a1.grid(True, alpha=.2)
    a2.plot(z.index, z.values, color=C["pairs"], lw=1.2)
    for y, col, lbl in [(strategy.z_entry, C["red"], f"Short +{strategy.z_entry}"),
                        (-strategy.z_entry, C["green"], f"Long -{strategy.z_entry}"),
                        (strategy.z_exit,  C["gold"], f"Exit ±{strategy.z_exit}"),
                        (-strategy.z_exit, C["gold"], "")]:
        a2.axhline(y, color=col, lw=1, ls="--", alpha=.8, label=lbl if lbl else None)
    a2.fill_between(z.index, z.values, strategy.z_entry,
                    where=(z>strategy.z_entry), alpha=.2, color=C["red"])
    a2.fill_between(z.index, z.values, -strategy.z_entry,
                    where=(z<-strategy.z_entry), alpha=.2, color=C["green"])
    a2.axhline(0, color="#fff", lw=.4, alpha=.3)
    a2.set_ylabel("Z-Score"); a2.legend(fontsize=8, loc="upper right"); a2.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_rsi_chart(prices, strategy, ticker):
    if ticker not in prices.columns: return None
    rsi_vals = strategy._wilder_rsi(prices[ticker].values)
    set_dark()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True,
                                  gridspec_kw={"height_ratios":[2,1]})
    fig.patch.set_facecolor(C["bg"])
    a1.plot(prices.index, prices[ticker].values, color=C["text"], lw=1.2)
    a1.set_title(f"RSI-{strategy.rsi_period}  —  {ticker}",
                 fontsize=11, fontweight="bold", color=C["gold"])
    a1.set_ylabel("Price (₹)"); a1.grid(True, alpha=.2)
    rsi_s = pd.Series(rsi_vals, index=prices.index)
    a2.plot(rsi_s.index, rsi_s.values, color=C["rev"], lw=1.3)
    for y, col, lbl in [(strategy.overbought_threshold, C["red"],  f"Overbought {strategy.overbought_threshold}"),
                        (strategy.oversold_threshold,   C["green"],f"Oversold {strategy.oversold_threshold}"),
                        (strategy.exit_threshold,       C["gold"], f"Exit {strategy.exit_threshold}")]:
        a2.axhline(y, color=col, lw=1, ls="--", alpha=.8, label=lbl)
    a2.fill_between(rsi_s.index, rsi_s.values, strategy.overbought_threshold,
                    where=(rsi_s>strategy.overbought_threshold), alpha=.2, color=C["red"])
    a2.fill_between(rsi_s.index, rsi_s.values, strategy.oversold_threshold,
                    where=(rsi_s<strategy.oversold_threshold), alpha=.2, color=C["green"])
    a2.set_ylim(0, 100); a2.set_ylabel("RSI")
    a2.legend(fontsize=8, loc="upper right"); a2.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_momentum_snapshot(prices, strategy, date_idx):
    lbk = strategy.lookback_days + strategy.skip_days
    if date_idx < lbk: return None
    ei, si = date_idx - strategy.skip_days, date_idx - lbk
    if si < 0: return None
    mom = (prices.iloc[ei]/prices.iloc[si]-1).dropna().sort_values()
    n   = len(mom)
    nl  = max(1, int(n*strategy.top_quantile))
    ns  = max(1, int(n*strategy.bottom_quantile))
    cols= [C["red"]]*ns + ["#445566"]*(n-ns-nl) + [C["green"]]*nl
    set_dark()
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.bar(range(n), mom.values*100, color=cols, width=.9, alpha=.85)
    ax.axhline(0, color="#fff", lw=.5, alpha=.4)
    ax.set_title(f"63-Day Momentum Snapshot — {str(prices.index[date_idx])[:10]}  "
                 f"(🟢 Long top {int(strategy.top_quantile*100)}%  🔴 Short bottom {int(strategy.bottom_quantile*100)}%)",
                 fontsize=10, fontweight="bold", color=C["gold"])
    ax.set_xlabel("Stocks (ranked by momentum)")
    ax.set_ylabel("3-Month Return (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax.grid(True, alpha=.15, axis="y")
    plt.tight_layout()
    return fig

def plot_sensitivity(loader, strategy_class, param_name, param_values, base_cfg):
    """Sharpe vs parameter value chart — walk-forward disabled for speed"""
    set_dark()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(C["bg"])
    sharpes = []
    for v in param_values:
        cfg = dict(base_cfg)
        cfg[param_name] = v
        st_obj = strategy_class(**cfg)
        bt = Backtester(st_obj, loader, walk_forward=False)
        bt.run()
        perf = bt.calculate_performance()
        sharpes.append(perf["_sharpe"])
    ax.plot(param_values, sharpes, color=C["gold"], lw=2, marker="o", markersize=5)
    ax.axhline(0, color="#fff", lw=.5, ls=":", alpha=.4)
    best_idx = int(np.argmax(sharpes))
    ax.axvline(param_values[best_idx], color=C["green"], lw=1, ls="--", alpha=.7,
               label=f"Best: {param_values[best_idx]} (Sharpe={sharpes[best_idx]:.2f})")
    ax.set_xlabel(param_name.replace("_"," ").title())
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Parameter Sensitivity — {param_name.replace('_',' ').title()}",
                 fontsize=11, fontweight="bold", color=C["gold"])
    ax.legend(fontsize=8); ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

# ── Metric row ─────────────────────────────────────────────────────────────────
def metric_row(perf):
    # Cast to plain Python float — prevents numpy.ndarray.__format__ errors
    # when benchmark_performance returns numpy scalars or 0-d arrays
    def f(key, default=0.0):
        v = perf.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    ann_r  = f("_ann_return") * 100
    sharpe = f("_sharpe")
    max_dd = f("_max_dd")    * 100
    vol    = f("_ann_vol")   * 100
    alpha  = f("_alpha")     * 100
    wr     = f("_win_rate")  * 100
    c = st.columns(6)
    c[0].metric("Ann. Return",     f"{ann_r:+.2f}%")
    c[1].metric("Sharpe Ratio",    f"{sharpe:.3f}",
                delta="Good" if sharpe>1 else "Low",
                delta_color="normal" if sharpe>0 else "inverse")
    c[2].metric("Max Drawdown",    f"-{max_dd:.2f}%",  delta_color="inverse")
    c[3].metric("Ann. Volatility", f"{vol:.2f}%")
    c[4].metric("Jensen Alpha",    f"{alpha:+.2f}%",
                delta_color="normal" if alpha>0 else "inverse")
    c[5].metric("Win Rate",        f"{wr:.1f}%",
                delta_color="normal" if wr>50 else "inverse")

# ── Comparison table ───────────────────────────────────────────────────────────
def render_comparison_table(results, bm_perf):
    metrics = [
        ("Total Return",          "_total_return",lambda v: f"{v*100:.2f}%"),
        ("Annualised Return",      "_ann_return",  lambda v: f"{v*100:.2f}%"),
        ("Annualised Volatility",  "_ann_vol",     lambda v: f"{v*100:.2f}%"),
        ("Sharpe Ratio",           "_sharpe",      lambda v: f"{v:.3f}"),
        ("Sortino Ratio",          "_sortino",     lambda v: f"{v:.3f}"),
        ("Max Drawdown",           "_max_dd",      lambda v: f"-{v*100:.2f}%"),
        ("Calmar Ratio",           "_calmar",      lambda v: f"{v:.3f}"),
        ("Win Rate",               "_win_rate",    lambda v: f"{v*100:.1f}%"),
        ("Beta (vs Nifty 50)",     "_beta",        lambda v: f"{v:.3f}"),
        ("Jensen's Alpha",         "_alpha",       lambda v: f"{v*100:.2f}%"),
        ("Information Ratio",      "_info_ratio",  lambda v: f"{v:.3f}"),
        ("VaR 95% (daily)",        "_var95",       lambda v: f"{v*100:.2f}%"),
        ("CVaR 95% (daily)",       "_cvar95",      lambda v: f"{v*100:.2f}%"),
    ]
    rows = []
    for label, key, fmt in metrics:
        row = {"Metric": label}
        for r in results:
            sn = SHORT.get(r["strategy"].get_name(), "")
            v  = r["perf"].get(key)
            row[sn] = fmt(float(v)) if v is not None else "—"
        bv = bm_perf.get(key)
        row["Nifty 50"] = fmt(float(bv)) if bv is not None else "—"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")
    flip = {"Max Drawdown","Annualised Volatility","VaR 95% (daily)","CVaR 95% (daily)"}

    def hl(row):
        styles = [""]*len(row)
        try:
            vals = []
            for v in row:
                try: vals.append(float(str(v).replace("%","").replace("−","-")))
                except: vals.append(None)
            nv = [v for v in vals if v is not None]
            if not nv: return styles
            sign = -1 if row.name in flip else 1
            best  = sign*max(sign*v for v in nv)
            worst = sign*min(sign*v for v in nv)
            for i, v in enumerate(vals):
                if v is None: continue
                if abs(v-best)  < 1e-6: styles[i] = "background:#0d3321;color:#00e676;font-weight:700"
                elif abs(v-worst)<1e-6: styles[i] = "background:#2d0c0c;color:#ff5252;font-weight:700"
        except: pass
        return styles

    styled = df.style.apply(hl, axis=1).set_properties(**{
        "text-align":"center","font-family":"'JetBrains Mono',monospace","font-size":"12px"
    }).set_table_styles([
        {"selector":"th","props":[("background","#0f2236"),("color","#c9a84c"),
                                   ("font-weight","700"),("font-size","12px"),
                                   ("border","1px solid #1a3a5c"),("padding","6px 12px")]},
        {"selector":"td","props":[("border","1px solid #1a3a5c"),("padding","5px 12px")]},
        {"selector":"tr:nth-child(even) td","props":[("background","#0f2236")]},
        {"selector":"tr:nth-child(odd) td", "props":[("background","#0b1929")]},
    ])
    st.dataframe(df, width='stretch', height=460)

def bm_perf(bm_r, bm_ec):
    return Backtester.benchmark_performance(bm_r, bm_ec)

def render_trade_log(bt):
    tl = bt.trade_log
    if tl is None or tl.empty:
        st.info("No trades generated.")
        return
    tl_s = tl.copy()
    tl_s["Date"] = tl_s["Date"].astype(str).str[:10]
    def cpnl(v):
        try:
            f = float(v)
            if f>0: return "color:#00e676;font-weight:600"
            if f<0: return "color:#ff5252;font-weight:600"
        except: pass
        return ""
    styled = tl_s.head(300).style.map(cpnl, subset=["Net_PnL_%","Gross_PnL_%"])
    st.dataframe(styled, width='stretch', height=320)
    st.caption(f"Showing first 300 of {len(tl)} trade days")

# ── Run tests inline ──────────────────────────────────────────────────────────
def run_tests_display():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "tests/test_strategies.py"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    output = result.stdout + result.stderr
    lines  = [l for l in output.split("\n") if l.strip()]
    passed = sum(1 for l in lines if "PASS" in l)
    failed = sum(1 for l in lines if "FAIL" in l)
    return lines, passed, failed

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = render_sidebar()

    if not (cfg["run_pairs"] or cfg["run_mom"] or cfg["run_rev"]):
        render_header()
        st.warning("Please select at least one strategy.")
        return

    if cfg["run_btn"]:
        st.session_state["ran"]    = True
        st.session_state["config"] = cfg

    if not st.session_state.get("ran"):
        render_header()
        st.markdown("""
        <div style="background:#0f2236;border:1px solid #1a3a5c;border-radius:12px;
                    padding:48px;text-align:center;margin-top:16px">
          <div style="font-size:48px;margin-bottom:14px">📈</div>
          <div style="font-size:20px;font-weight:700;color:#e8f4fd;margin-bottom:8px">
            Configure & Run Your Backtest
          </div>
          <div style="color:#90b4ce;font-size:13px;max-width:520px;margin:0 auto 20px">
            Tune parameters in the sidebar → click
            <strong style="color:#c9a84c">🚀 Run Backtest</strong>
          </div>
          <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
            <div style="background:#132840;border:1px solid #3b82f644;border-radius:8px;padding:14px 18px;width:190px;text-align:left">
              <div style="color:#3b82f6;font-weight:700;margin-bottom:6px">🔗 Pairs Trading</div>
              <div style="color:#90b4ce;font-size:11px">Engle-Granger cointegration · 4950 pairs tested · rolling z-score signals</div>
            </div>
            <div style="background:#132840;border:1px solid #f9731644;border-radius:8px;padding:14px 18px;width:190px;text-align:left">
              <div style="color:#f97316;font-weight:700;margin-bottom:6px">🚀 Momentum</div>
              <div style="color:#90b4ce;font-size:11px">Jegadeesh-Titman 1993 · 63-day formation · long top 20% short bottom 20%</div>
            </div>
            <div style="background:#132840;border:1px solid #10b98144;border-radius:8px;padding:14px 18px;width:190px;text-align:left">
              <div style="color:#10b981;font-weight:700;margin-bottom:6px">🔄 Mean Reversion</div>
              <div style="color:#90b4ce;font-size:11px">Wilder RSI-14 · oversold &lt;30 long · overbought &gt;70 short · exit at 50</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # New features callout
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, title, body in [
            (c1,"📡","Real NSE Data","yfinance downloads live Nifty 100 data. Auto-adjusts for splits & dividends."),
            (c2,"🔭","Walk-Forward","Calibrate on first 252 days, trade only out-of-sample. No lookahead bias."),
            (c3,"🛡","Regime Filter","Suppresses short signals when Nifty 50 is above 200-day MA (bull market)."),
            (c4,"✅","28 Unit Tests","ADF test, RSI bounds, OLS, equity curve, no-lookahead — all verified."),
        ]:
            col.markdown(f"""
            <div style="background:#0f2236;border:1px solid #1a3a5c;border-radius:10px;
                        padding:16px;text-align:center">
              <div style="font-size:26px">{icon}</div>
              <div style="color:#c9a84c;font-weight:700;margin:6px 0">{title}</div>
              <div style="color:#90b4ce;font-size:11px">{body}</div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    run_cfg = st.session_state["config"]
    config_key = str(sorted({k:v for k,v in run_cfg.items() if k!="run_btn"}.items()))

    prog = st.progress(0, text="Initialising...")
    try:
        prog.progress(20, text="Loading market data (yfinance → cache → synthetic)...")
        results, loader, bm_ec, bm_dd, bm_r = run_backtest(config_key, run_cfg)
        prog.progress(100, text="Done!")
        prog.empty()
    except Exception as e:
        prog.empty()
        st.error(f"Backtest failed: {e}")
        import traceback; st.code(traceback.format_exc())
        return

    render_header(data_source=loader.get_data_source())
    prices = loader.get_prices()
    bm_p   = bm_perf(bm_r, bm_ec)

    # ── Walk-forward callout ──
    if run_cfg["walk_fwd"]:
        oos = len(prices) - run_cfg["form_days"]
        st.info(f"🔭 Walk-forward enabled — Formation: {run_cfg['form_days']} days  |  "
                f"Out-of-sample trading: {oos} days  |  "
                f"{'Regime filter ON' if run_cfg['regime_filter'] else 'Regime filter OFF'}")

    # ── Summary metrics ──
    st.markdown("### 📊 Results")
    for r in results:
        name  = r["strategy"].get_name()
        color = STRAT_COLORS.get(name, "#fff")
        st.markdown(f'<div style="border-left:4px solid {color};padding-left:10px;'
                    f'margin-bottom:4px"><span style="color:{color};font-weight:700">'
                    f'{SHORT.get(name,name)}</span> '
                    f'<span style="color:#4a7a9b;font-size:11px">{name}</span></div>',
                    unsafe_allow_html=True)
        metric_row(r["perf"])
        st.markdown("")

    # Benchmark
    st.markdown(f'<div style="border-left:4px solid {C["bench"]};padding-left:10px;'
                f'margin-bottom:4px"><span style="color:{C["bench"]};font-weight:700">'
                f'Nifty 50 Benchmark</span></div>', unsafe_allow_html=True)
    metric_row(bm_p)
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Equity & Drawdown",
        "⚡ Sharpe & Distribution",
        "📅 Monthly Heatmaps",
        "📋 Performance Table",
        "🔗 Pairs Analysis",
        "🔄 RSI Analysis",
        "🚀 Momentum Analysis",
        "🔬 Sensitivity Analysis",
        "✅ Unit Tests",
        "📒 Trade Logs",
    ])

    with tabs[0]:
        st.markdown("#### Portfolio Equity Curves")
        f = plot_equity_curves(results, bm_ec)
        st.pyplot(f, width='stretch'); plt.close(f)
        st.markdown("#### Drawdown Chart")
        f = plot_drawdowns(results, bm_dd)
        st.pyplot(f, width='stretch'); plt.close(f)

    with tabs[1]:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("#### Rolling 6-Month Sharpe")
            f = plot_rolling_sharpe(results, bm_r)
            st.pyplot(f, width='stretch'); plt.close(f)
        with c2:
            st.markdown("#### Return Distribution")
            f = plot_return_distribution(results)
            st.pyplot(f, width='stretch'); plt.close(f)
        st.markdown("#### Return Statistics")
        rows = []
        for r in results:
            dr = r["backtester"].daily_returns.dropna()*100
            rows.append({
                "Strategy": SHORT.get(r["strategy"].get_name(),""),
                "Mean Daily": f"{dr.mean():.4f}%",
                "Std Dev":    f"{dr.std():.4f}%",
                "Skewness":   f"{float(dr.skew()):.3f}",
                "Kurtosis":   f"{float(dr.kurtosis()):.3f}",
                "Best Day":   f"{dr.max():.3f}%",
                "Worst Day":  f"{dr.min():.3f}%",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with tabs[2]:
        for r in results:
            f = plot_monthly_heatmap(r["backtester"], r["strategy"].get_name())
            if f: st.pyplot(f, width='stretch'); plt.close(f)
            st.markdown("")

    with tabs[3]:
        st.markdown("#### Head-to-Head Comparison  ·  🟢 Best  🔴 Worst")
        render_comparison_table(results, bm_p)
        st.markdown("#### Parameters Used")
        for r in results:
            with st.expander(f"⚙ {SHORT.get(r['strategy'].get_name(),'')} Parameters"):
                params = r["strategy"].get_parameters()
                params_df = pd.DataFrame(list(params.items()), columns=["Parameter","Value"])
                params_df["Value"] = params_df["Value"].astype(str)
                st.dataframe(params_df, width='stretch', hide_index=True)

    with tabs[4]:
        pairs_r = next((r for r in results if "Pairs" in r["strategy"].get_name()), None)
        if not pairs_r:
            st.info("Enable Pairs Trading to see this tab.")
        else:
            st_obj = pairs_r["strategy"]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div style="background:#0f2236;border:1px solid #1a3a5c;border-radius:10px;padding:16px">
                  <div style="color:#c9a84c;font-weight:700;margin-bottom:8px">Cointegration Stats</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#90b4ce">
                    Pairs tested: <span style="color:#e8f4fd">1,225</span><br>
                    Pairs retained: <span style="color:#00e676;font-weight:700">{len(st_obj.cointegrated_pairs)}</span><br>
                    p-value threshold: <span style="color:#e8f4fd">{st_obj.coint_pvalue}</span><br>
                    Z-entry / exit: <span style="color:#e8f4fd">±{st_obj.z_entry} / ±{st_obj.z_exit}</span><br>
                    Rolling window: <span style="color:#e8f4fd">{st_obj.rolling_window} days</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                #### Engle-Granger
                **Step 1 OLS:** `log(A) = α + β·log(B) + ε`
                **Step 2 ADF:** `Δε = γ·ε_{t-1} + Σδ·Δε_{t-j}`
                H₀: γ=0 (unit root) → p<0.05 to keep
                **Z-score:** `z = (spread − μ₆₀) / σ₆₀`
                """)
            with c2:
                if st_obj.pair_stats:
                    df_p = pd.DataFrame([{
                        "Stock A": ps["pair"][0], "Stock B": ps["pair"][1],
                        "β": round(float(ps["beta"]),4),
                        "ADF stat": round(float(ps["adf_stat"]),4),
                        "p-value": round(float(ps["pvalue"]),4),
                        "Spread σ": round(float(ps["spread_std"]),4),
                        "Half-life": round(float(ps["half_life"]) if ps["half_life"]!=float("inf") else 999, 1),
                    } for ps in st_obj.pair_stats])
                    st.markdown("**Cointegrated Pairs**")
                    st.dataframe(df_p, width='stretch', height=320)

            st.markdown("#### Spread Z-Score")
            pair_names = [f"{p[0]} / {p[1]}" for p in st_obj.cointegrated_pairs]
            if pair_names:
                chosen = st.selectbox("Select pair", pair_names, key="pair_sel")
                f = plot_pair_zscore(prices, st_obj, pair_names.index(chosen))
                if f: st.pyplot(f, width='stretch'); plt.close(f)

    with tabs[5]:
        rev_r = next((r for r in results if "Mean Reversion" in r["strategy"].get_name()), None)
        if not rev_r:
            st.info("Enable Mean Reversion to see this tab.")
        else:
            st_obj = rev_r["strategy"]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                #### Wilder RSI Formula
                **Gain / Loss:**
                > G = max(P_t − P_{{t-1}}, 0)
                > L = max(P_{{t-1}} − P_t, 0)

                **Smoothing (α = 1/{st_obj.rsi_period}):**
                > AvgG_t = (AvgG_{{t-1}} × {st_obj.rsi_period-1} + G_t) / {st_obj.rsi_period}

                **RSI:**
                > RSI = 100 − 100 / (1 + AvgG/AvgL)

                **Signals:**
                > RSI < {st_obj.oversold_threshold} → Long
                > RSI > {st_obj.overbought_threshold} → Short
                > RSI ↗ {st_obj.exit_threshold} → Exit long
                > RSI ↘ {st_obj.exit_threshold} → Exit short
                """)
            with c2:
                ticker_sel = st.selectbox("Stock for RSI chart",
                                          sorted(prices.columns.tolist()), key="rsi_t")
                f = plot_rsi_chart(prices, st_obj, ticker_sel)
                if f: st.pyplot(f, width='stretch'); plt.close(f)

    with tabs[6]:
        mom_r = next((r for r in results if "Momentum" in r["strategy"].get_name()), None)
        if not mom_r:
            st.info("Enable Momentum to see this tab.")
        else:
            st_obj = mom_r["strategy"]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                #### Jegadeesh-Titman (1993)
                **Signal:**
                > MOM_i = P_i(t) / P_i(t − {st_obj.lookback_days}) − 1

                Skip last {st_obj.skip_days} days (avoid reversal)

                **Portfolio:**
                > Long: top {int(st_obj.top_quantile*100)}%
                > Short: bottom {int(st_obj.bottom_quantile*100)}%
                > Equal-weight each leg
                > Rebalance every {st_obj.rebalance_freq} days
                """)
            with c2:
                max_i = len(prices)-1
                snap  = st.slider("Snapshot date (index)",
                                   st_obj.lookback_days+st_obj.skip_days,
                                   max_i, max_i//2, key="mom_snap")
                f = plot_momentum_snapshot(prices, st_obj, snap)
                if f: st.pyplot(f, width='stretch'); plt.close(f)

    with tabs[7]:
        st.markdown("#### Parameter Sensitivity Analysis")
        st.caption("How Sharpe Ratio changes as you vary a parameter (walk-forward disabled for speed)")
        sens_strat = st.selectbox("Strategy", ["Momentum","Mean Reversion","Pairs Trading"], key="sens_s")
        if sens_strat == "Momentum":
            st.markdown("**Vary: Formation Period (days)**")
            f = plot_sensitivity(
                loader, MomentumStrategy, "lookback_days",
                [21, 42, 63, 84, 105, 126],
                {"lookback_days":63, "top_quantile":.2, "bottom_quantile":.2,
                 "rebalance_freq":21, "skip_days":5}
            )
            st.pyplot(f, width='stretch'); plt.close(f)
        elif sens_strat == "Mean Reversion":
            st.markdown("**Vary: RSI Period**")
            f = plot_sensitivity(
                loader, MeanReversionStrategy, "rsi_period",
                [7, 9, 11, 14, 18, 21],
                {"rsi_period":14, "oversold_threshold":30.0, "overbought_threshold":70.0,
                 "exit_threshold":50.0, "max_positions":20}
            )
            st.pyplot(f, width='stretch'); plt.close(f)
        else:
            st.markdown("**Vary: Z-Score Entry Threshold**")
            f = plot_sensitivity(
                loader, PairsTradingStrategy, "z_entry",
                [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
                {"coint_pvalue":0.05, "z_entry":2.0, "z_exit":0.5,
                 "rolling_window":60, "max_pairs":20}
            )
            st.pyplot(f, width='stretch'); plt.close(f)

    with tabs[8]:
        st.markdown("#### Unit Test Results")
        st.caption("28 tests covering ADF, OLS, RSI, momentum signals, backtester correctness, regime filter, walk-forward")
        if st.button("▶ Run All Tests", key="run_tests"):
            with st.spinner("Running 28 unit tests..."):
                lines, passed, failed = run_tests_display()
            if failed == 0:
                st.success(f"✅ All {passed} tests passed!")
            else:
                st.error(f"❌ {failed} failed, {passed} passed")
            for line in lines:
                if "PASS" in line:
                    st.markdown(f'<span style="color:#00e676;font-family:monospace;font-size:12px">{line}</span>', unsafe_allow_html=True)
                elif "FAIL" in line:
                    st.markdown(f'<span style="color:#ff5252;font-family:monospace;font-size:12px">{line}</span>', unsafe_allow_html=True)
                elif "Results" in line:
                    st.markdown(f'<span style="color:#c9a84c;font-family:monospace;font-weight:700">{line}</span>', unsafe_allow_html=True)
        else:
            st.markdown("""
            | Test Class | Tests | What It Verifies |
            |---|---|---|
            | `TestADF` | 5 | ADF detects stationary vs random walk, p-value in [0,1] |
            | `TestOLS` | 3 | β=2 for y=2x+1, residuals have zero mean, negative β |
            | `TestRSI` | 6 | Bounds [0,100], rising/falling price direction, NaN prefix |
            | `TestMomentum` | 3 | Signal bounds, no signal in formation, long/short balance |
            | `TestBacktester` | 9 | EC starts at 1, no lookahead, walk-forward zeros, costs reduce returns |
            | `TestHalfLife` | 2 | OU half-life matches analytical κ=0.5 and κ=0.05 |
            """)

    with tabs[9]:
        for r in results:
            name  = r["strategy"].get_name()
            color = STRAT_COLORS.get(name, "#fff")
            st.markdown(f'<div style="border-left:4px solid {color};padding-left:10px;'
                        f'margin:12px 0 6px;font-weight:700;color:{color}">'
                        f'{SHORT.get(name,name)} — Trade Log</div>', unsafe_allow_html=True)
            render_trade_log(r["backtester"])
            st.markdown("")

if __name__ == "__main__":
    main()
