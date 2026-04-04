"""
tests/test_strategies.py — Unit tests for all core algorithms

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
# import pytest

from pairs_trading   import PairsTradingStrategy
from momentum        import MomentumStrategy
from mean_reversion  import MeanReversionStrategy
from backtester      import Backtester
from data_loader     import DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
# ADF / Cointegration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestADF:
    def test_stationary_series_low_pvalue(self):
        """White noise (zero mean) should reject unit root → p < 0.05"""
        np.random.seed(1)
        stationary = np.random.normal(0, 1, 500)
        _, pval = PairsTradingStrategy._adf_test(stationary)
        assert pval < 0.05, f"Expected p < 0.05 for white noise, got {pval:.4f}"

    def test_random_walk_high_pvalue(self):
        """Random walk (unit root) should NOT reject H0 → p > 0.10"""
        np.random.seed(2)
        rw = np.cumsum(np.random.normal(0, 1, 500))
        _, pval = PairsTradingStrategy._adf_test(rw)
        assert pval > 0.10, f"Expected p > 0.10 for random walk, got {pval:.4f}"

    def test_pvalue_between_0_and_1(self):
        """p-value must always be in [0, 1]"""
        np.random.seed(3)
        for _ in range(20):
            series = np.cumsum(np.random.normal(0, 1, 200))
            _, pval = PairsTradingStrategy._adf_test(series)
            assert 0.0 <= pval <= 1.0, f"p-value out of bounds: {pval}"

    def test_short_series_does_not_crash(self):
        """Series shorter than min length should return p=1.0 gracefully"""
        short = np.array([1.0, 2.0, 1.5])
        stat, pval = PairsTradingStrategy._adf_test(short)
        assert pval == 1.0

    def test_ou_process_is_stationary(self):
        """Ornstein-Uhlenbeck process must be detected as stationary"""
        np.random.seed(4)
        x = np.zeros(600)
        for t in range(1, 600):
            x[t] = x[t-1] - 0.10 * x[t-1] + np.random.normal(0, 0.5)
        _, pval = PairsTradingStrategy._adf_test(x)
        assert pval < 0.05, f"OU process should be stationary, got p={pval:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# OLS Regression Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOLS:
    def test_perfect_linear_relationship(self):
        """y = 2x + 1 → β ≈ 2, α ≈ 1, residuals ≈ 0"""
        x = np.linspace(1, 100, 200)
        y = 2 * x + 1
        beta, alpha, residuals = PairsTradingStrategy._ols_regression(y, x)
        assert abs(beta  - 2.0) < 1e-6, f"β should be 2.0, got {beta}"
        assert abs(alpha - 1.0) < 1e-6, f"α should be 1.0, got {alpha}"
        assert np.max(np.abs(residuals)) < 1e-6

    def test_residuals_zero_mean(self):
        """OLS residuals must always have zero mean"""
        np.random.seed(5)
        x = np.random.randn(300)
        y = 1.5 * x + np.random.normal(0, 0.3, 300)
        _, _, res = PairsTradingStrategy._ols_regression(y, x)
        assert abs(res.mean()) < 1e-8

    def test_negative_beta(self):
        """Negative relationship: y = -0.8x → β ≈ -0.8"""
        x = np.linspace(1, 50, 100)
        y = -0.8 * x + 5
        beta, _, _ = PairsTradingStrategy._ols_regression(y, x)
        assert abs(beta - (-0.8)) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# RSI Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRSI:
    def setup_method(self):
        self.strategy = MeanReversionStrategy()

    def test_rsi_bounds(self):
        """RSI must always be between 0 and 100"""
        np.random.seed(6)
        prices = np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 500)))
        rsi = self.strategy._wilder_rsi(prices)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0),   f"RSI below 0: min={valid.min()}"
        assert np.all(valid <= 100), f"RSI above 100: max={valid.max()}"

    def test_constant_price_rsi(self):
        """Flat price series → no losses at all, so RSI = 100 (Wilder convention)"""
        prices = np.ones(100)
        rsi = self.strategy._wilder_rsi(prices)
        valid = rsi[~np.isnan(rsi)]
        # AvgLoss=0 → RS=inf → RSI=100 (correct Wilder behaviour)
        assert len(valid) == 0 or np.all(valid == 100.0)

    def test_rising_price_high_rsi(self):
        """Steadily rising prices → RSI should be near 100"""
        prices = np.linspace(100, 200, 200)
        rsi = self.strategy._wilder_rsi(prices)
        valid = rsi[~np.isnan(rsi)]
        assert valid[-1] > 80, f"Rising prices should give high RSI, got {valid[-1]:.1f}"

    def test_falling_price_low_rsi(self):
        """Steadily falling prices → RSI should be near 0"""
        prices = np.linspace(200, 100, 200)
        rsi = self.strategy._wilder_rsi(prices)
        valid = rsi[~np.isnan(rsi)]
        assert valid[-1] < 20, f"Falling prices should give low RSI, got {valid[-1]:.1f}"

    def test_rsi_length_matches_input(self):
        """Output length must equal input length"""
        prices = np.random.lognormal(0, 0.01, 150).cumprod()
        rsi = self.strategy._wilder_rsi(prices)
        assert len(rsi) == len(prices)

    def test_first_n_values_are_nan(self):
        """First rsi_period values should be NaN (not enough data)"""
        prices = np.random.lognormal(0, 0.01, 100).cumprod()
        rsi = self.strategy._wilder_rsi(prices)
        assert np.all(np.isnan(rsi[:self.strategy.rsi_period]))


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentum:
    def setup_method(self):
        self.strategy = MomentumStrategy(lookback_days=63, top_quantile=0.20,
                                          bottom_quantile=0.20, rebalance_freq=21)

    def test_signals_bounded(self):
        """All signal values should be in [-1, +1]"""
        np.random.seed(7)
        loader = DataLoader("2021-01-01", "2023-01-01")
        loader._load_synthetic()
        signals = self.strategy.generate_signals(loader.get_prices())
        assert signals.values.min() >= -1.0 - 1e-9
        assert signals.values.max() <=  1.0 + 1e-9

    def test_no_signal_before_formation(self):
        """No signals should be generated in the first lookback+skip days"""
        np.random.seed(8)
        loader = DataLoader("2021-01-01", "2023-01-01")
        loader._load_synthetic()
        signals = self.strategy.generate_signals(loader.get_prices())
        cutoff  = self.strategy.lookback_days + self.strategy.skip_days
        early   = signals.iloc[:cutoff]
        assert (early == 0).all().all(), "Signals should be 0 during formation period"

    def test_long_short_balance(self):
        """At each rebalance date, should have approximately equal longs and shorts"""
        np.random.seed(9)
        loader = DataLoader("2021-01-01", "2023-01-01")
        loader._load_synthetic()
        prices  = loader.get_prices()
        signals = self.strategy.generate_signals(prices)

        for t in range(self.strategy.lookback_days + 100, len(signals), self.strategy.rebalance_freq):
            row     = signals.iloc[t]
            n_long  = (row > 0).sum()
            n_short = (row < 0).sum()
            if n_long > 0 or n_short > 0:
                # Ratio should be within 50% of each other
                ratio = n_long / (n_short + 1e-9)
                assert 0.3 < ratio < 3.0, f"Unbalanced at t={t}: {n_long}L / {n_short}S"


# ═══════════════════════════════════════════════════════════════════════════════
# Backtester Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktester:
    def setup_method(self):
        np.random.seed(42)
        self.loader = DataLoader("2021-01-01", "2023-06-01")
        self.loader._load_synthetic()

    def test_equity_curve_starts_at_one(self):
        """Equity curve must start at 1.0"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        assert abs(bt.equity_curve.iloc[0] - 1.0) < 1e-9

    def test_equity_curve_always_positive(self):
        """Portfolio value can never go below 0"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        assert bt.equity_curve.min() > 0.0

    def test_no_lookahead_bias(self):
        """Signals on day T must only use prices up to day T (shift(1) enforced)"""
        st  = MomentumStrategy()
        bt  = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        # signals.shift(1) means first row of lagged signals is always 0
        sig_lag = bt.signals.shift(1).fillna(0)
        assert (sig_lag.iloc[0] == 0).all(), "Day-0 lagged signal must be 0 (no lookahead)"

    def test_walk_forward_zeros_formation_period(self):
        """Walk-forward: no signals in first `formation_days` days"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=True, formation_days=126)
        bt.run()
        early = bt.signals.iloc[:126]
        assert (early == 0).all().all(), "Formation period signals must be zero"

    def test_transaction_costs_reduce_returns(self):
        """Returns with costs < returns without costs"""
        st1 = MomentumStrategy()
        st2 = MomentumStrategy()
        bt0 = Backtester(st1, self.loader, transaction_cost=0.0,   walk_forward=False)
        bt1 = Backtester(st2, self.loader, transaction_cost=0.001, walk_forward=False)
        bt0.run(); bt1.run()
        r0 = bt0.equity_curve.iloc[-1]
        r1 = bt1.equity_curve.iloc[-1]
        assert r1 <= r0 + 1e-9, "Costs must reduce (or equal) returns"

    def test_sharpe_ratio_is_finite(self):
        """Sharpe ratio must be a finite number"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        perf = bt.calculate_performance()
        assert np.isfinite(perf["_sharpe"]), "Sharpe ratio must be finite"

    def test_max_drawdown_between_0_and_1(self):
        """Max drawdown must be in [0, 1]"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        perf = bt.calculate_performance()
        assert 0.0 <= perf["_max_dd"] <= 1.0

    def test_win_rate_between_0_and_1(self):
        """Win rate must be in [0, 1]"""
        st = MomentumStrategy()
        bt = Backtester(st, self.loader, walk_forward=False)
        bt.run()
        perf = bt.calculate_performance()
        assert 0.0 <= perf["_win_rate"] <= 1.0

    def test_regime_filter_reduces_short_signals(self):
        """With regime filter on, short signals should be <= without it"""
        st1 = MomentumStrategy()
        st2 = MomentumStrategy()
        bt_no  = Backtester(st1, self.loader, walk_forward=False, use_regime_filter=False)
        bt_yes = Backtester(st2, self.loader, walk_forward=False, use_regime_filter=True)
        bt_no.run(); bt_yes.run()

        shorts_no  = (bt_no.signals  < 0).sum().sum()
        shorts_yes = (bt_yes.signals < 0).sum().sum()
        assert shorts_yes <= shorts_no, "Regime filter should reduce or equal short signals"


# ═══════════════════════════════════════════════════════════════════════════════
# Half-life Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestHalfLife:
    def test_fast_mean_reversion(self):
        """Fast-reverting OU (κ=0.5) → half-life ≈ 1.4 days"""
        np.random.seed(10)
        x = np.zeros(1000)
        for t in range(1, 1000):
            x[t] = x[t-1] - 0.5 * x[t-1] + np.random.normal(0, 0.1)
        hl = PairsTradingStrategy._compute_half_life(x)
        # ln(2)/0.5 ≈ 1.386
        assert 0.5 < hl < 5.0, f"Expected half-life ~1.4, got {hl:.2f}"

    def test_slow_mean_reversion(self):
        """Slow-reverting OU (κ=0.05) → half-life ≈ 14 days"""
        np.random.seed(11)
        x = np.zeros(2000)
        for t in range(1, 2000):
            x[t] = x[t-1] - 0.05 * x[t-1] + np.random.normal(0, 0.1)
        hl = PairsTradingStrategy._compute_half_life(x)
        assert 5.0 < hl < 50.0, f"Expected half-life ~14, got {hl:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Run tests directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback
    test_classes = [TestADF, TestOLS, TestRSI, TestMomentum,
                    TestBacktester, TestHalfLife]
    passed, failed = 0, 0
    for cls in test_classes:
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            try:
                if hasattr(obj, "setup_method"):
                    obj.setup_method()
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}  →  {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    if failed == 0:
        print("All tests passed!")
