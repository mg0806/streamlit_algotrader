"""
MomentumStrategy — Jegadeesh-Titman (1993) Cross-Sectional Momentum Factor.

Algorithm:
  - At each monthly rebalance date:
    1. Calculate 63-trading-day (3-month) total return for every stock
       R_i = (P_{t} / P_{t-63}) - 1
    2. Rank all stocks by R_i (ascending)
    3. Long top 20% (quintile 5, momentum winners)
    4. Short bottom 20% (quintile 1, momentum losers)
    5. Equal-weight within long and short legs
    6. Hold for 1 month, then rebalance

Reference: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency", Journal of Finance.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from base_strategy import Strategy


class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum — long winners, short losers.

    The momentum signal for stock i at time t:
      MOM_i(t) = P_i(t) / P_i(t - L) - 1

    where L = lookback_days (default 63 = ~3 months).

    Portfolio construction:
      - Sort all stocks by MOM_i
      - Long: top Q fraction (default 0.20)
      - Short: bottom Q fraction (default 0.20)
      - Equal weight within each leg
      - Rebalance every R days (default 21 = ~1 month)
    """

    def __init__(self, lookback_days: int = 63, top_quantile: float = 0.20,
                 bottom_quantile: float = 0.20, rebalance_freq: int = 21,
                 skip_days: int = 5):
        """
        Parameters
        ----------
        lookback_days   : Formation period in trading days (63 ≈ 3 months)
        top_quantile    : Fraction of stocks to long (top momentum)
        bottom_quantile : Fraction of stocks to short (bottom momentum)
        rebalance_freq  : Rebalance frequency in trading days (21 ≈ 1 month)
        skip_days       : Skip the most recent N days to avoid short-term reversal
                          (standard in academic momentum: skip last 5 days)
        """
        self.lookback_days    = lookback_days
        self.top_quantile     = top_quantile
        self.bottom_quantile  = bottom_quantile
        self.rebalance_freq   = rebalance_freq
        self.skip_days        = skip_days

    def get_name(self) -> str:
        return 'Cross-Sectional Momentum (Jegadeesh-Titman)'

    def get_parameters(self) -> dict:
        return {
            'lookback_period_days':    self.lookback_days,
            'skip_recent_days':        self.skip_days,
            'top_quantile_long':       f'Top {int(self.top_quantile*100)}%',
            'bottom_quantile_short':   f'Bottom {int(self.bottom_quantile*100)}%',
            'rebalance_frequency':     f'Every {self.rebalance_freq} trading days',
            'position_sizing':         'Equal weight within each leg',
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals at each rebalance date.

        Returns DataFrame with same shape as prices:
          +1/n_long  = long position (normalised)
          -1/n_short = short position (normalised)
          0          = no position
        """
        print(f"  [{self.get_name()}]")
        print(f"  Lookback: {self.lookback_days}d | "
              f"Top/Bottom: {int(self.top_quantile*100)}% | "
              f"Rebalance: every {self.rebalance_freq}d")

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_stocks = len(prices.columns)
        n_long   = max(1, int(n_stocks * self.top_quantile))
        n_short  = max(1, int(n_stocks * self.bottom_quantile))

        # Formation starts after lookback_days + skip_days
        start_idx = self.lookback_days + self.skip_days

        rebalance_dates = []
        current_signals = pd.Series(0.0, index=prices.columns)
        n_rebalances = 0

        for t in range(start_idx, len(prices)):
            # Rebalance on schedule
            if (t - start_idx) % self.rebalance_freq == 0:
                # Momentum return: skip most recent `skip_days` to avoid reversal
                # R_i = P(t - skip_days) / P(t - skip_days - lookback_days) - 1
                end_idx   = t - self.skip_days
                start_lbk = end_idx - self.lookback_days

                if start_lbk < 0:
                    continue

                price_end   = prices.iloc[end_idx]
                price_start = prices.iloc[start_lbk]

                # Compute total return over formation period
                momentum_returns = (price_end / price_start) - 1.0

                # Remove NaN
                valid = momentum_returns.dropna()
                if len(valid) < 10:
                    continue

                # Rank and assign signals
                ranked = valid.rank(ascending=True)
                n_valid = len(ranked)
                n_l = max(1, int(n_valid * self.top_quantile))
                n_s = max(1, int(n_valid * self.bottom_quantile))

                current_signals = pd.Series(0.0, index=prices.columns)

                # Long: top n_l stocks (highest momentum)
                winners = ranked[ranked >= (n_valid - n_l + 1)].index
                for w in winners:
                    current_signals[w] = 1.0 / n_l

                # Short: bottom n_s stocks (lowest momentum)
                losers = ranked[ranked <= n_s].index
                for l in losers:
                    current_signals[l] = -1.0 / n_s

                rebalance_dates.append(prices.index[t])
                n_rebalances += 1

            signals.iloc[t] = current_signals

        print(f"  Rebalanced {n_rebalances} times | "
              f"~{n_long} longs + {n_short} shorts per period")
        return signals
