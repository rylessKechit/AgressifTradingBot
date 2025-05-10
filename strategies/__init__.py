"""
Module de stratégies de trading
Contient différentes stratégies de trading pour le bot
"""

from strategies.base_strategy import BaseStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.combined_strategy import CombinedStrategy

__all__ = ['BaseStrategy', 'TrendFollowingStrategy', 'MeanReversionStrategy', 
          'ArbitrageStrategy', 'CombinedStrategy']