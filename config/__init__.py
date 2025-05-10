"""
Module de configuration
Contient les paramètres et la configuration globale du bot de trading
"""

from config.settings import *
from config.strategy_params import *

__all__ = ['TRADING', 'CAPITAL', 'EXCHANGE', 'BACKTEST', 'NOTIFICATIONS', 'EXECUTION_MODE',
          'TREND_FOLLOWING_PARAMS', 'MEAN_REVERSION_PARAMS', 'ARBITRAGE_PARAMS', 'COMBINED_STRATEGY_PARAMS',
          'RISK_MANAGEMENT_PARAMS']