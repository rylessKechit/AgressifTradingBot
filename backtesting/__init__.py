"""
Module de backtesting
Contient les classes et fonctions pour tester les stratégies de trading sur des données historiques
"""

from backtesting.optimizer import StrategyOptimizer
from backtesting.performance import PerformanceAnalyzer

__all__ = ['StrategyOptimizer', 'PerformanceAnalyzer']