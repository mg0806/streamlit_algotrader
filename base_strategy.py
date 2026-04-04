"""
Strategy — Abstract base class that all trading strategies must inherit from.
Defines the interface: generate_signals(), get_name(), get_parameters().
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Any new strategy can be added to the framework by:
      1. Subclassing Strategy
      2. Implementing generate_signals(), get_name(), get_parameters()

    Signal convention:
      +1  = Long position
      -1  = Short position
       0  = Flat (no position)
    """

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a DataFrame of position signals aligned with the price DataFrame.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily closing prices, indexed by date, columns = ticker symbols.

        Returns
        -------
        pd.DataFrame
            Same shape as prices. Values in {-1, 0, +1}.
            signals.iloc[t][ticker] = position to hold at close of day t.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable strategy name string."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return a dictionary of strategy parameters for logging and display."""
        pass
