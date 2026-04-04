"""
MeanReversionStrategy — RSI-Based Mean Reversion.

Algorithm:
  For each stock, compute 14-day RSI daily:
    RSI_t = 100 - 100 / (1 + RS_t)
    RS_t  = AvgGain_t / AvgLoss_t  (Wilder smoothed averages)

  Signals:
    RSI < 30 (oversold)  → Long  (enter)
    RSI > 50             → Exit long
    RSI > 70 (overbought)→ Short (enter)
    RSI < 50             → Exit short

Position sizing: equal weight across all active signals.

Reference: Wilder, J.W. (1978) "New Concepts in Technical Trading Systems"
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from base_strategy import Strategy


class MeanReversionStrategy(Strategy):
    """
    RSI Mean Reversion — buy oversold, sell overbought.

    RSI calculation (Wilder smoothing):
      Initial AvgGain = mean of gains over first `period` days
      Initial AvgLoss = mean of losses over first `period` days
      Subsequent:
        AvgGain_t = (AvgGain_{t-1} * (period-1) + Gain_t) / period
        AvgLoss_t = (AvgLoss_{t-1} * (period-1) + Loss_t) / period

      RS_t = AvgGain_t / AvgLoss_t
      RSI_t = 100 - 100 / (1 + RS_t)
    """

    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30.0,
                 overbought_threshold: float = 70.0, exit_threshold: float = 50.0,
                 max_positions: int = 20):
        """
        Parameters
        ----------
        rsi_period            : RSI lookback period (Wilder default = 14)
        oversold_threshold    : RSI < this → buy signal (default 30)
        overbought_threshold  : RSI > this → sell signal (default 70)
        exit_threshold        : RSI crosses this to exit position (default 50)
        max_positions         : Max simultaneous positions (to limit capital)
        """
        self.rsi_period            = rsi_period
        self.oversold_threshold    = oversold_threshold
        self.overbought_threshold  = overbought_threshold
        self.exit_threshold        = exit_threshold
        self.max_positions         = max_positions

    def get_name(self) -> str:
        return 'Mean Reversion (RSI-14)'

    def get_parameters(self) -> dict:
        return {
            'rsi_period':             self.rsi_period,
            'oversold_entry':         f'RSI < {self.oversold_threshold}',
            'overbought_entry':       f'RSI > {self.overbought_threshold}',
            'exit_signal':            f'RSI crosses {self.exit_threshold}',
            'max_simultaneous_pos':   self.max_positions,
            'position_sizing':        'Equal weight across active signals',
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI for all stocks and generate signals.

        Returns DataFrame of signals (values in {-1, 0, +1}).
        Positions are normalised by number of active signals.
        """
        print(f"  [{self.get_name()}]")
        print(f"  RSI({self.rsi_period}) | "
              f"Oversold <{self.oversold_threshold} | "
              f"Overbought >{self.overbought_threshold} | "
              f"Exit at {self.exit_threshold}")

        # Step 1: Compute RSI for every stock
        rsi_df = self._compute_rsi_all(prices)

        # Step 2: Generate raw entry/exit signals from RSI
        signals = self._rsi_to_signals(rsi_df)

        n_long  = (signals > 0).sum().sum()
        n_short = (signals < 0).sum().sum()
        print(f"  Total long signals: {n_long} | Total short signals: {n_short}")
        return signals

    # ── RSI computation ───────────────────────────────────────────────────

    def _compute_rsi_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute Wilder RSI for all stocks efficiently using vectorized ops."""
        rsi_dict = {}
        for ticker in prices.columns:
            rsi_dict[ticker] = self._wilder_rsi(prices[ticker].values)
        return pd.DataFrame(rsi_dict, index=prices.index)

    def _wilder_rsi(self, price_series: np.ndarray) -> np.ndarray:
        """
        Wilder smoothed RSI.

        Gain_t = max(P_t - P_{t-1}, 0)
        Loss_t = max(P_{t-1} - P_t, 0)

        Seed (simple average for first `period` obs):
          AvgGain_period = mean(Gain_1..Gain_period)
          AvgLoss_period = mean(Loss_1..Loss_period)

        Subsequent (Wilder EMA with α = 1/period):
          AvgGain_t = (AvgGain_{t-1} * (n-1) + Gain_t) / n
          AvgLoss_t = (AvgLoss_{t-1} * (n-1) + Loss_t) / n
        """
        n = len(price_series)
        rsi = np.full(n, np.nan)
        if n < self.rsi_period + 1:
            return rsi

        deltas = np.diff(price_series)
        gains  = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)

        p = self.rsi_period
        # Seed
        avg_gain = gains[:p].mean()
        avg_loss = losses[:p].mean()

        if avg_loss < 1e-12:
            rsi[p] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[p] = 100.0 - 100.0 / (1.0 + rs)

        # Wilder smoothing
        for i in range(p, n - 1):
            avg_gain = (avg_gain * (p - 1) + gains[i]) / p
            avg_loss = (avg_loss * (p - 1) + losses[i]) / p
            if avg_loss < 1e-12:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    # ── Signal generation from RSI ─────────────────────────────────────────

    def _rsi_to_signals(self, rsi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert RSI values to position signals with state machine logic.

        States per stock:
          position[ticker] = +1 (long), -1 (short), 0 (flat)

        Transitions:
          flat  → long  : RSI crosses below oversold_threshold
          flat  → short : RSI crosses above overbought_threshold
          long  → flat  : RSI crosses above exit_threshold (50)
          short → flat  : RSI crosses below exit_threshold (50)
        """
        signals = pd.DataFrame(0.0, index=rsi_df.index, columns=rsi_df.columns)

        for ticker in rsi_df.columns:
            rsi    = rsi_df[ticker].values
            pos    = 0  # current position
            sig    = np.zeros(len(rsi))

            for t in range(1, len(rsi)):
                r_prev = rsi[t - 1]
                r_curr = rsi[t]
                if np.isnan(r_curr):
                    sig[t] = 0
                    continue

                # Entry conditions
                if pos == 0:
                    if r_prev >= self.oversold_threshold and r_curr < self.oversold_threshold:
                        pos = 1   # enter long (RSI crossed below 30)
                    elif r_prev <= self.overbought_threshold and r_curr > self.overbought_threshold:
                        pos = -1  # enter short (RSI crossed above 70)

                # Exit conditions
                elif pos == 1:  # long position
                    if r_prev <= self.exit_threshold and r_curr > self.exit_threshold:
                        pos = 0   # exit long (RSI recovered above 50)

                elif pos == -1:  # short position
                    if r_prev >= self.exit_threshold and r_curr < self.exit_threshold:
                        pos = 0   # exit short (RSI fell below 50)

                sig[t] = pos

            signals[ticker] = sig

        # Normalise: at each time t, divide by number of active positions
        n_active = (signals != 0).sum(axis=1).clip(lower=1)
        # Cap max positions
        for t in range(len(signals)):
            row = signals.iloc[t]
            active_idx = row[row != 0].index
            if len(active_idx) > self.max_positions:
                # Keep max_positions randomly (in practice: rank by RSI extremity)
                keep = active_idx[:self.max_positions]
                drop = active_idx[self.max_positions:]
                signals.iloc[t][drop] = 0.0

        # Normalise by active count for equal weighting
        n_active = (signals != 0).sum(axis=1).replace(0, 1)
        signals = signals.div(n_active, axis=0)

        return signals
