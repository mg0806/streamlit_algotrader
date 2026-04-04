"""
backtester.py — Production-grade Backtester (v3 — fully numpy-safe)

All arithmetic on returns and benchmark uses numpy arrays extracted
upfront. Zero pandas Series-vs-Series arithmetic anywhere in metrics,
which eliminates all index-alignment errors across pandas versions.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class Backtester:
    RISK_FREE_RATE = 0.065
    TRADING_DAYS   = 252

    def __init__(self, strategy, data_loader,
                 transaction_cost=0.001,
                 walk_forward=True,
                 formation_days=252,
                 use_regime_filter=True):
        self.strategy          = strategy
        self.data_loader       = data_loader
        self.transaction_cost  = transaction_cost
        self.walk_forward      = walk_forward
        self.formation_days    = formation_days
        self.use_regime_filter = use_regime_filter

        self.signals       = None
        self.equity_curve  = None
        self.daily_returns = None
        self.trade_log     = None
        self._performance  = None

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        prices  = self.data_loader.get_prices()
        returns = self.data_loader.get_returns()
        bm_r    = self.data_loader.get_benchmark_returns()

        print(f"\n{'='*60}")
        print(f"Running: {self.strategy.get_name()}")
        if self.walk_forward:
            print(f"  Walk-forward: formation={self.formation_days}d "
                  f"| trading={len(prices)-self.formation_days}d")
        print(f"{'='*60}")

        self.signals = self.strategy.generate_signals(prices)

        if self.walk_forward and self.formation_days > 0:
            self.signals.iloc[:self.formation_days] = 0.0

        if self.use_regime_filter:
            self.signals = self._apply_regime_filter(self.signals, bm_r)

        signals_lag  = self.signals.shift(1).fillna(0)
        total_abs    = signals_lag.abs().sum(axis=1).replace(0, 1)
        signals_norm = signals_lag.div(total_abs, axis=0)

        gross_pnl = (signals_norm * returns).sum(axis=1)
        costs     = signals_norm.diff().abs().sum(axis=1) * self.transaction_cost

        net_returns        = gross_pnl - costs
        net_returns.name   = self.strategy.get_name()
        self.equity_curve  = (1 + net_returns).cumprod()
        self.daily_returns = net_returns
        self._signals_norm = signals_norm

        self.trade_log = self._build_trade_log(signals_norm, returns, costs)
        total = (self.equity_curve.iloc[-1] - 1) * 100
        print(f"  Done — Net return: {total:.1f}%")

        return {"equity_curve": self.equity_curve,
                "daily_returns": self.daily_returns,
                "signals": self.signals,
                "trade_log": self.trade_log}

    # ── Regime filter — pure numpy, no pandas arithmetic ─────────────────────

    def _apply_regime_filter(self, signals, bm_r):
        # Align benchmark to signal dates, extract as 1-D float array
        bm_vals  = bm_r.reindex(signals.index).fillna(0).values.astype(float).ravel()
        bm_cum   = np.cumprod(1.0 + bm_vals)          # cumulative price index

        # Rolling 200-day mean — numpy loop (T ≈ 1500, fast enough)
        window    = 200
        bull_mask = np.ones(len(bm_cum), dtype=bool)   # default True
        for t in range(window, len(bm_cum)):
            bull_mask[t] = bm_cum[t] > bm_cum[t - window:t].mean()

        # signals → 2-D float array, apply suppression, rebuild DataFrame
        sig_arr  = signals.values.astype(float).copy()           # (T, N)
        suppress = (sig_arr < 0) & bull_mask.reshape(-1, 1)      # (T, N) bool
        sig_arr[suppress] = 0.0

        n = int(suppress.sum())
        if n > 0:
            print(f"  Regime filter: suppressed {n} short signals in bull market")

        return pd.DataFrame(sig_arr,
                            index=signals.index,
                            columns=signals.columns)

    # ── Performance metrics — pure numpy ─────────────────────────────────────

    def calculate_performance(self) -> dict:
        # Align portfolio and benchmark on common dates, extract numpy arrays
        r_series  = self.daily_returns.dropna()
        bm_raw    = self.data_loader.get_benchmark_returns()
        common    = r_series.index.intersection(bm_raw.index)
        r_series  = r_series.loc[common]
        bm_series = bm_raw.loc[common].fillna(0)

        r  = r_series.values.astype(float).ravel()    # guaranteed (T,)
        bm = bm_series.values.astype(float).ravel()   # guaranteed (T,)
        T  = len(r)
        if T < 2:
            raise ValueError("Not enough trading days to calculate performance.")

        rf_d = float((1.0 + self.RISK_FREE_RATE) ** (1.0 / self.TRADING_DAYS) - 1.0)

        # Returns
        total  = float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)
        ann_r  = float((1.0 + total) ** (self.TRADING_DAYS / T) - 1.0)
        ann_v  = float(r.std()) * float(np.sqrt(self.TRADING_DAYS))
        sharpe = float((r.mean() - rf_d) / (r.std() + 1e-12) * np.sqrt(self.TRADING_DAYS))

        dside   = r[r < rf_d] - rf_d
        ds_std  = float(np.sqrt((dside**2).mean()) * np.sqrt(self.TRADING_DAYS)) if len(dside) > 0 else 1e-12
        sortino = float((ann_r - self.RISK_FREE_RATE) / (ds_std + 1e-12))

        max_dd, dd_dur = self._max_drawdown(self.equity_curve)
        calmar   = float(ann_r / (abs(max_dd) + 1e-12))

        active   = r[r != 0]
        wins     = active[active > 0]
        losses   = active[active < 0]
        win_rate = float(len(wins) / (len(active) + 1e-12))
        avg_wl   = float(wins.mean() / (abs(losses.mean()) + 1e-12)) if len(losses) > 0 else float("inf")
        pf       = float(wins.sum()  / (abs(losses.sum())  + 1e-12)) if len(losses) > 0 else float("inf")

        # Beta / Jensen Alpha
        bm_dm  = bm - bm.mean()
        r_dm   = r  - r.mean()
        beta   = float(np.dot(bm_dm, r_dm) / (np.dot(bm_dm, bm_dm) + 1e-12))
        bm_ann = float((1.0 + bm.mean()) ** self.TRADING_DAYS - 1.0)
        j_alpha= float(ann_r - (self.RISK_FREE_RATE + beta * (bm_ann - self.RISK_FREE_RATE)))

        # Information Ratio
        act_r  = r - bm
        info_r = float(act_r.mean() / (act_r.std() + 1e-12) * np.sqrt(self.TRADING_DAYS))

        # VaR / CVaR
        var95  = float(-np.percentile(r, 5))
        tail   = r[r < -var95]
        cvar95 = float(-tail.mean()) if len(tail) > 0 else var95

        self._performance = {
            "Strategy":                 self.strategy.get_name(),
            "Total Return":             f"{total*100:.2f}%",
            "Annualised Return":        f"{ann_r*100:.2f}%",
            "Annualised Volatility":    f"{ann_v*100:.2f}%",
            "Sharpe Ratio":             f"{sharpe:.3f}",
            "Sortino Ratio":            f"{sortino:.3f}",
            "Maximum Drawdown":         f"{max_dd*100:.2f}%",
            "Drawdown Duration (days)": f"{int(dd_dur)}",
            "Calmar Ratio":             f"{calmar:.3f}",
            "Win Rate":                 f"{win_rate*100:.1f}%",
            "Avg Win / Avg Loss":       f"{avg_wl:.2f}",
            "Profit Factor":            f"{pf:.2f}",
            "Beta (vs Nifty 50)":       f"{beta:.3f}",
            "Jensen's Alpha":           f"{j_alpha*100:.2f}%",
            "Information Ratio":        f"{info_r:.3f}",
            "VaR (95%, daily)":         f"{var95*100:.2f}%",
            "CVaR / ES (95%)":          f"{cvar95*100:.2f}%",
            "_ann_return":  ann_r,   "_ann_vol":    ann_v,
            "_sharpe":      sharpe,  "_sortino":    sortino,
            "_max_dd":      max_dd,  "_dd_duration":dd_dur,
            "_calmar":      calmar,  "_win_rate":   win_rate,
            "_beta":        beta,    "_alpha":      j_alpha,
            "_info_ratio":  info_r,  "_var95":      var95,
            "_cvar95":      cvar95,  "_total_return":total,
        }
        return self._performance

    # ── Rolling metrics ───────────────────────────────────────────────────────

    def rolling_sharpe(self, window=126) -> pd.Series:
        r        = self.daily_returns
        rf_daily = float((1.0 + self.RISK_FREE_RATE) ** (1.0 / self.TRADING_DAYS) - 1.0)
        excess_s = pd.Series(r.values.astype(float) - rf_daily, index=r.index)
        roll_m   = excess_s.rolling(window).mean()
        roll_s   = r.rolling(window).std()
        return (roll_m / (roll_s + 1e-12) * np.sqrt(self.TRADING_DAYS))

    def monthly_returns(self) -> pd.DataFrame:
        monthly = self.daily_returns.resample("ME").apply(lambda x: (1+x).prod()-1)
        monthly.index = monthly.index.to_period("M")
        pivot = monthly.to_frame("return")
        pivot["Year"]  = pivot.index.year
        pivot["Month"] = pivot.index.month
        return pivot.pivot(index="Year", columns="Month", values="return")

    def drawdown_series(self) -> pd.Series:
        peak = self.equity_curve.cummax()
        return (self.equity_curve - peak) / peak

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(ec):
        vals  = ec.values.astype(float)
        peak  = vals[0]; max_dd = 0.0; max_dur = 0; dur = 0
        for v in vals:
            if v > peak: peak = v; dur = 0
            else:        dur += 1
            dd = (peak - v) / (peak + 1e-12)
            if dd > max_dd:   max_dd  = dd
            if dur > max_dur: max_dur = dur
        return float(max_dd), int(max_dur)

    @staticmethod
    def _beta_alpha(rp, rm):
        rp = np.asarray(rp, dtype=float)
        rm = np.asarray(rm, dtype=float)
        rm_dm = rm - rm.mean()
        rp_dm = rp - rp.mean()
        beta  = float(np.dot(rm_dm, rp_dm) / (np.dot(rm_dm, rm_dm) + 1e-12))
        alpha = float(rp.mean() - beta * rm.mean())
        return beta, alpha

    def _build_trade_log(self, signals_norm, returns, costs):
        rows, ret_idx = [], returns.index
        for t in range(1, len(signals_norm)):
            date = signals_norm.index[t]
            if date not in ret_idx: continue
            row = signals_norm.iloc[t]
            if (row == 0).all(): continue
            rpos  = ret_idx.get_loc(date)
            gross = float((row * returns.iloc[rpos]).sum())
            cost  = float(costs.iloc[t])
            rows.append({
                "Date":               date,
                "N_Long":             int((row > 0).sum()),
                "N_Short":            int((row < 0).sum()),
                "Gross_PnL_%":        round(gross * 100, 4),
                "Transaction_Cost_%": round(cost  * 100, 4),
                "Net_PnL_%":          round((gross - cost) * 100, 4),
                "Portfolio_Value":    round(float(self.equity_curve.iloc[t]), 6),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def benchmark_performance(bm_r, bm_ec):
        r    = bm_r.dropna().values.astype(float).ravel()
        ec_v = bm_ec.values.astype(float).ravel()
        T    = len(r)
        rf   = 0.065
        rf_d = float((1 + rf) ** (1/252) - 1)

        total   = float(ec_v[-1] / ec_v[0] - 1)
        ann_r   = float((1 + total) ** (252/T) - 1)
        ann_v   = float(r.std() * np.sqrt(252))
        sharpe  = float((r.mean() - rf_d) / (r.std() + 1e-9) * np.sqrt(252))
        dside   = r[r < rf_d] - rf_d
        sortino = float((ann_r - rf) / (np.sqrt((dside**2).mean()) * np.sqrt(252) + 1e-9)) \
                  if len(dside) > 0 else 0.0

        pk = ec_v[0]; max_dd = 0.0
        for v in ec_v:
            if v > pk: pk = v
            dd = (pk - v) / (pk + 1e-9)
            if dd > max_dd: max_dd = dd
        max_dd = float(max_dd)
        calmar = float(ann_r / (max_dd + 1e-9))

        wins  = r[r > 0]
        wr    = float(len(wins) / (len(r[r != 0]) + 1e-9))
        var95 = float(-np.percentile(r, 5))
        tail  = r[r < -var95]
        cvar95= float(-tail.mean()) if len(tail) > 0 else var95

        return {
            "_ann_return":  ann_r,   "_ann_vol":   ann_v,
            "_sharpe":      sharpe,  "_sortino":   sortino,
            "_max_dd":      max_dd,  "_calmar":    calmar,
            "_win_rate":    wr,      "_beta":      1.0,
            "_alpha":       0.0,     "_info_ratio":0.0,
            "_var95":       var95,   "_cvar95":    cvar95,
            "_total_return":total,   "_dd_duration":0,
        }
