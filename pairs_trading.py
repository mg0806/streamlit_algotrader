"""
PairsTradingStrategy — Engle-Granger Two-Step Cointegration Pairs Trading.

Algorithm:
  1. Test all N*(N-1)/2 stock pairs for cointegration using Engle-Granger.
  2. Retain pairs where ADF p-value on residuals < 0.05.
  3. For each cointegrated pair:
     a. Estimate hedge ratio β via OLS: log(A_t) = α + β*log(B_t) + ε_t
     b. Spread_t = log(A_t) - β*log(B_t)
     c. Compute rolling 60-day z-score of spread
     d. Long spread (long A, short B) when z < -2.0
     e. Short spread (short A, long B) when z > +2.0
     f. Exit when |z| < 0.5
"""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from base_strategy import Strategy


class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage using Engle-Granger cointegration test.

    The Engle-Granger two-step procedure:
      Step 1: Estimate OLS  y_t = α + β*x_t + ε_t
      Step 2: Run ADF test on residuals ε̂_t
              If ADF p-value < threshold → series are cointegrated

    ADF test (Augmented Dickey-Fuller):
      Δε_t = α + γ*ε_{t-1} + Σ_{j=1}^{p} δ_j * Δε_{t-j} + u_t
      H0: γ = 0 (unit root / non-stationary)
      H1: γ < 0 (stationary / mean-reverting)

    Z-score of spread:
      z_t = (spread_t - μ_{60}) / σ_{60}
    where μ and σ are rolling 60-day mean and standard deviation.
    """

    def __init__(self, coint_pvalue: float = 0.05, z_entry: float = 2.0,
                 z_exit: float = 0.5, rolling_window: int = 60,
                 max_pairs: int = 50):
        self.coint_pvalue   = coint_pvalue
        self.z_entry        = z_entry
        self.z_exit         = z_exit
        self.rolling_window = rolling_window
        self.max_pairs      = max_pairs
        self.cointegrated_pairs = []
        self.hedge_ratios   = {}
        self.pair_stats     = []

    def get_name(self) -> str:
        return 'Pairs Trading (Engle-Granger Cointegration)'

    def get_parameters(self) -> dict:
        return {
            'cointegration_pvalue_threshold': self.coint_pvalue,
            'z_score_entry_threshold':        self.z_entry,
            'z_score_exit_threshold':         self.z_exit,
            'rolling_z_window_days':          self.rolling_window,
            'max_pairs_traded':               self.max_pairs,
        }

    # ── Signal generation ─────────────────────────────────────────────────

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Main entry: find cointegrated pairs then generate daily signals."""
        log_prices = np.log(prices)

        # ── Step 1: Use first 252 days as formation period ──
        formation_end = min(252, len(log_prices) // 3)
        formation_prices = log_prices.iloc[:formation_end]

        print(f"  [{self.get_name()}]")
        print(f"  Testing {len(list(combinations(prices.columns[:50], 2)))} pairs "
              f"(top 50 stocks) for cointegration...")

        self._find_cointegrated_pairs(formation_prices, prices.columns[:50].tolist())
        print(f"  Found {len(self.cointegrated_pairs)} cointegrated pairs "
              f"(p < {self.coint_pvalue})")

        if not self.cointegrated_pairs:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # ── Step 2: Generate signals using rolling z-score ──
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals = self._generate_pair_signals(log_prices, signals)
        return signals

    # ── Cointegration testing ─────────────────────────────────────────────

    def _find_cointegrated_pairs(self, log_prices: pd.DataFrame, universe: list):
        """
        Engle-Granger cointegration test for all pairs in universe.
        Keeps pairs with ADF p-value < self.coint_pvalue on the OLS residuals.
        """
        self.cointegrated_pairs = []
        self.hedge_ratios = {}
        self.pair_stats = []

        all_pairs = list(combinations(universe, 2))
        scores = []

        for stockA, stockB in all_pairs:
            if stockA not in log_prices.columns or stockB not in log_prices.columns:
                continue
            y = log_prices[stockA].values
            x = log_prices[stockB].values

            # OLS regression: y = α + β*x + ε
            beta, alpha, residuals = self._ols_regression(y, x)

            # ADF test on residuals
            adf_stat, pvalue = self._adf_test(residuals)

            if pvalue < self.coint_pvalue:
                spread_std   = np.std(residuals)
                half_life    = self._compute_half_life(residuals)
                scores.append({
                    'pair':       (stockA, stockB),
                    'beta':       beta,
                    'alpha':      alpha,
                    'adf_stat':   adf_stat,
                    'pvalue':     pvalue,
                    'spread_std': spread_std,
                    'half_life':  half_life,
                })

        # Sort by p-value, take best max_pairs pairs
        scores.sort(key=lambda x: x['pvalue'])
        scores = scores[:self.max_pairs]

        for s in scores:
            pair = s['pair']
            self.cointegrated_pairs.append(pair)
            self.hedge_ratios[pair]  = s['beta']
            self.pair_stats.append(s)

    # ── Signal generation for each pair ───────────────────────────────────

    def _generate_pair_signals(self, log_prices: pd.DataFrame,
                                signals: pd.DataFrame) -> pd.DataFrame:
        """
        For each cointegrated pair, compute rolling z-score and generate signals.

        Signal logic:
          z_t > +entry  → Short spread: signal_A = -1, signal_B = +1
          z_t < -entry  → Long spread:  signal_A = +1, signal_B = -1
          |z_t| < exit  → Flat:         signal_A =  0, signal_B =  0
        """
        pair_position = {p: 0 for p in self.cointegrated_pairs}

        # Accumulate signals — normalise by number of active pairs
        raw_signals = {ticker: pd.Series(0.0, index=log_prices.index)
                       for ticker in log_prices.columns}

        for pair in self.cointegrated_pairs:
            stockA, stockB = pair
            if stockA not in log_prices.columns or stockB not in log_prices.columns:
                continue

            beta  = self.hedge_ratios[pair]
            y     = log_prices[stockA]
            x     = log_prices[stockB]
            spread = y - beta * x

            # Rolling z-score
            roll_mean = spread.rolling(self.rolling_window, min_periods=30).mean()
            roll_std  = spread.rolling(self.rolling_window, min_periods=30).std()
            z_score   = (spread - roll_mean) / (roll_std + 1e-9)

            pos = 0  # current position: +1 long spread, -1 short spread, 0 flat
            sig_a = pd.Series(0.0, index=log_prices.index)
            sig_b = pd.Series(0.0, index=log_prices.index)

            for t in range(self.rolling_window, len(z_score)):
                z = z_score.iloc[t]
                if np.isnan(z):
                    continue
                # Entry signals
                if pos == 0:
                    if z < -self.z_entry:
                        pos = 1   # long spread: long A, short B
                    elif z > self.z_entry:
                        pos = -1  # short spread: short A, long B
                # Exit signals
                elif abs(z) < self.z_exit:
                    pos = 0
                sig_a.iloc[t] = pos
                sig_b.iloc[t] = -pos

            raw_signals[stockA] = raw_signals[stockA] + sig_a
            raw_signals[stockB] = raw_signals[stockB] + sig_b

        # Normalise: cap positions at ±1
        for ticker in log_prices.columns:
            if ticker in raw_signals:
                s = raw_signals[ticker]
                signals[ticker] = np.sign(s).where(s != 0, 0)

        return signals

    # ── Statistical helper functions ──────────────────────────────────────

    @staticmethod
    def _ols_regression(y: np.ndarray, x: np.ndarray):
        """
        OLS regression: y = α + β*x + ε
        β = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
        α = ȳ - β*x̄
        """
        x_dm = x - x.mean()
        y_dm = y - y.mean()
        beta  = np.dot(x_dm, y_dm) / (np.dot(x_dm, x_dm) + 1e-12)
        alpha = y.mean() - beta * x.mean()
        residuals = y - alpha - beta * x
        return beta, alpha, residuals

    @staticmethod
    def _adf_test(series: np.ndarray, max_lags: int = 4) -> tuple:
        """
        Augmented Dickey-Fuller test (implemented from scratch).

        Model: Δy_t = α + γ*y_{t-1} + Σ_{j=1}^{p} δ_j*Δy_{t-j} + ε_t

        Test statistic: t = γ̂ / SE(γ̂)
        Null: γ = 0 (unit root)
        Alternative: γ < 0 (stationary)

        Critical values (MacKinnon 1994) for constant model:
          1%: -3.43,  5%: -2.86,  10%: -2.57
        P-value approximated using response surface coefficients.
        """
        n = len(series)
        if n < 20:
            return 0.0, 1.0

        dy = np.diff(series)
        y_lag = series[:-1]

        # Build regressor matrix: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}, 1]
        p = min(max_lags, n // 10)
        nobs = len(dy) - p

        Y = dy[p:]
        X = np.column_stack([
            y_lag[p:],
            *[dy[p-j:-j if j > 0 else None] for j in range(1, p+1)],
            np.ones(nobs)
        ])

        # OLS: β = (X'X)^{-1} X'Y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta_hat = XtX_inv @ X.T @ Y
            residuals = Y - X @ beta_hat
            s2 = np.dot(residuals, residuals) / (nobs - X.shape[1])
            se_gamma = np.sqrt(s2 * XtX_inv[0, 0])
            t_stat = beta_hat[0] / (se_gamma + 1e-12)
        except Exception:
            return 0.0, 1.0

        # MacKinnon (1994) approximate p-value for n_obs observations
        # Response surface: p = Φ(τ) adjusted for finite samples
        p_value = PairsTradingStrategy._adf_pvalue(t_stat, n)
        return t_stat, p_value

    @staticmethod
    def _adf_pvalue(t_stat: float, nobs: int) -> float:
        """
        Approximate ADF p-value using MacKinnon (2010) response surface.
        Tabulated critical values for 'c' (constant only) model:
          p=0.01: -3.43  p=0.025: -3.12  p=0.05: -2.86
          p=0.10: -2.57  p=0.20: -2.23
        Linear interpolation between tabulated values.
        """
        # Finite-sample adjustment
        adj = 1.0 + 8.5 / nobs
        t_adj = t_stat / adj

        # MacKinnon percentile/t-stat mapping
        percentiles = [0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 1.00]
        crit_vals   = [-3.43, -3.12, -2.86, -2.57, -2.23, -1.53, -0.67,  0.0, 2.0]

        if t_adj <= crit_vals[0]:
            return 0.001
        if t_adj >= crit_vals[-1]:
            return 0.999
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= t_adj < crit_vals[i+1]:
                w = (t_adj - crit_vals[i]) / (crit_vals[i+1] - crit_vals[i])
                return percentiles[i] + w * (percentiles[i+1] - percentiles[i])
        return 0.5

    @staticmethod
    def _compute_half_life(residuals: np.ndarray) -> float:
        """
        Compute mean-reversion half-life from OU process.
        Δε_t = κ*(μ - ε_{t-1}) + σ*dW
        Estimate κ via OLS: Δε_t = a + b*ε_{t-1}
        Half-life = -ln(2) / b
        """
        dy = np.diff(residuals)
        y_lag = residuals[:-1]
        y_lag_dm = y_lag - y_lag.mean()
        dy_dm    = dy - dy.mean()
        b = np.dot(y_lag_dm, dy_dm) / (np.dot(y_lag_dm, y_lag_dm) + 1e-12)
        if b >= 0:
            return np.inf
        return -np.log(2) / b
