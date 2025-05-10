"""
Script d'optimisation optimale pour crypto
Optimisé pour atteindre 10% de rendement mensuel avec un risque maîtrisé
"""
import os
import sys
import time
import json
import logging
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les modules nécessaires
from config.settings import BACKTEST, TRADING, CAPITAL
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.combined_strategy import CombinedStrategy
from execution.trader import Trader
from risk.position_sizing import PositionSizer
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger

# Configuration du logging
logger = setup_logger(level=logging.INFO)

class OptimalCryptoOptimizer:
    """
    Classe pour l'optimisation optimale des stratégies de trading crypto
    Spécialisée pour atteindre 10% mensuel avec un risque maîtrisé
    """
    
    def __init__(self, target_monthly_return=10.0, max_iterations=40, population_size=30, 
                 symbol="BTC/USDT", timeframe="15m", start_date=None, end_date=None,
                 strategy_name="combined", results_dir="optimal_crypto_results",
                 max_acceptable_drawdown=25.0, risk_per_trade=0.02, stagnation_limit=5):
        """
        Initialise l'optimiseur de crypto optimal
        
        Args:
            target_monthly_return (float): Rendement mensuel cible en pourcentage
            max_iterations (int): Nombre maximum d'itérations
            population_size (int): Taille de la population par itération
            symbol (str): Symbole de la paire à trader
            timeframe (str): Timeframe
            start_date (str): Date de début (format YYYY-MM-DD)
            end_date (str): Date de fin (format YYYY-MM-DD)
            strategy_name (str): Nom de la stratégie ("trend", "mean_reversion", "combined")
            results_dir (str): Répertoire pour sauvegarder les résultats
            max_acceptable_drawdown (float): Drawdown maximum acceptable en pourcentage
            risk_per_trade (float): Pourcentage du capital à risquer par trade
            stagnation_limit (int): Nombre d'itérations sans amélioration avant arrêt
        """
        self.target_monthly_return = target_monthly_return
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.risk_per_trade = risk_per_trade
        self.stagnation_limit = stagnation_limit
        
        # Convertir les dates si fournies
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = datetime.now() - timedelta(days=90)  # Par défaut: 3 mois
            
        if end_date:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = datetime.now()
            
        self.strategy_name = strategy_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Historique des optimisations
        self.optimization_history = []
        self.stagnation_counter = 0
        self.previous_best_fitness = -float('inf')
        
        # Initialiser les composants
        self.fetcher = DataFetcher()
        
        # Créer un position sizer équilibré
        self.position_sizer = PositionSizer(
            initial_capital=CAPITAL.get('initial_capital', 10000),
            max_risk_per_trade=risk_per_trade,  # 2% par trade
            max_position_size=0.3  # 30% du capital max
        )
        
        # Espaces de paramètres pour chaque stratégie (optimisés pour crypto)
        self.param_spaces = self._init_crypto_param_spaces()
        
        # Calculer la durée du test en mois pour normaliser le rendement
        delta = self.end_date - self.start_date
        self.test_duration_months = delta.days / 30.44  # Approximation du nombre de mois
        
        # Calculer le rendement cible total pour la période
        self.target_total_return = (1 + target_monthly_return/100) ** self.test_duration_months * 100 - 100
        
        logger.info(f"Optimiseur crypto initialisé avec cible de rendement mensuel de {target_monthly_return}%")
        logger.info(f"Pour la période de test, cela correspond à un rendement total cible de {self.target_total_return:.2f}%")
        logger.info(f"Stratégie: {strategy_name}, Paire: {symbol}, Timeframe: {timeframe}")
        logger.info(f"Période: {self.start_date.strftime('%Y-%m-%d')} à {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Drawdown maximum acceptable: {max_acceptable_drawdown}%")
        logger.info(f"Risque par trade: {risk_per_trade*100}%")
        
    def _init_crypto_param_spaces(self):
        """
        Initialise les espaces de paramètres optimaux pour crypto
        Basé sur des recherches et stratégies éprouvées pour la crypto
        
        Returns:
            dict: Espaces de paramètres
        """
        param_spaces = {
            "trend": {
                # Périodes des moyennes mobiles optimisées pour crypto
                "fast_ema": (6, 10),          # Optimisé pour crypto
                "medium_ema": (18, 24),       # Optimisé pour crypto
                "slow_ema": (45, 55),         # Optimisé pour crypto
                "very_slow_ema": (190, 210),  # Optimisé pour crypto
                
                # MACD (paramètres optimisés pour crypto)
                "macd_fast": (10, 13),        # Optimisé pour crypto
                "macd_slow": (24, 28),        # Optimisé pour crypto
                "macd_signal": (8, 10),       # Optimisé pour crypto
                
                # ADX (optimisé pour crypto)
                "adx_threshold": (20, 28),    # Optimisé pour crypto
                
                # Seuils de signaux (optimisés pour réduire le bruit)
                "buy_threshold": (0.4, 0.65),  # Optimisé pour réduire les faux signaux
                "sell_threshold": (-0.65, -0.4), # Optimisé pour réduire les faux signaux
                
                # Poids
                "weight": (0.45, 0.55)          # Équilibré
            },
            "mean_reversion": {
                # RSI (optimisé pour crypto)
                "rsi_period": (12, 16),        # Optimisé pour crypto
                "rsi_overbought": (68, 73),    # Optimisé pour crypto
                "rsi_oversold": (27, 32),      # Optimisé pour crypto
                
                # Bandes de Bollinger (optimisé pour crypto)
                "bb_period": (18, 22),         # Optimisé pour crypto
                "bb_std": (1.9, 2.2),          # Optimisé pour crypto
                
                # Stochastique (optimisé pour crypto)
                "stoch_k": (12, 16),           # Optimisé pour crypto
                "stoch_d": (3, 4),             # Optimisé pour crypto
                "stoch_overbought": (75, 80),  # Optimisé pour crypto
                "stoch_oversold": (20, 25),    # Optimisé pour crypto
                
                # Seuils de signaux
                "buy_threshold": (0.4, 0.65),  # Optimisé pour réduire les faux signaux
                "sell_threshold": (-0.65, -0.4), # Optimisé pour réduire les faux signaux
                
                # Poids
                "weight": (0.45, 0.55)          # Équilibré
            },
            "combined": {
                # Seuils finaux (optimisés pour 10% mensuel)
                "final_buy_threshold": (0.3, 0.45),   # Optimisé pour 10% mensuel
                "final_sell_threshold": (-0.45, -0.3), # Optimisé pour 10% mensuel
                
                # Filtres (optimisés pour crypto)
                "adjust_weights_by_volatility": [True],  # Toujours activé
                "use_volume_filter": [True],             # Toujours activé
                "volume_threshold": (1.5, 2.2),          # Optimisé pour crypto
                "use_multi_timeframe": [True],           # Toujours activé
                "cooldown_period": (2, 4),               # Optimisé pour 15m
                
                # Timeframe (optimisé pour meilleure performance)
                "timeframe_alignment_required": [False],  # Plus flexible
            }
        }
        
        return param_spaces
    
    def _calculate_fitness(self, result):
        """
        Calcule le score de fitness d'un résultat
        Optimisé pour 10% mensuel avec gestion du risque améliorée
        
        Args:
            result (dict): Résultat du backtest
            
        Returns:
            float: Score de fitness
        """
        profit_loss_pct = result.get('profit_loss_pct', 0)
        max_drawdown = result.get('max_drawdown', 1)
        win_rate = result.get('win_rate', 0)
        total_trades = result.get('total_trades', 0)
        sharpe_ratio = result.get('sharpe_ratio', 0)
        
        # Paramètres cibles pour 10% mensuel
        ideal_monthly_return = self.target_monthly_return
        ideal_drawdown = 10.0  # Drawdown idéal de 10% maximum
        ideal_win_rate = 60.0  # Taux de réussite idéal d'au moins 60%
        ideal_trades_per_month = 100  # Nombre idéal de trades par mois
        
        # Calculer le rendement mensuel
        if self.test_duration_months > 0:
            monthly_return = profit_loss_pct / self.test_duration_months
            trades_per_month = total_trades / self.test_duration_months
        else:
            monthly_return = profit_loss_pct
            trades_per_month = total_trades
        
        # Pénalités pour drawdown excessif
        if max_drawdown * 100 > self.max_acceptable_drawdown:
            drawdown_penalty = (max_drawdown * 100 - self.max_acceptable_drawdown) * 2
        else:
            # Bonus pour drawdown inférieur à l'idéal
            if max_drawdown * 100 < ideal_drawdown:
                drawdown_penalty = -5 * (ideal_drawdown - max_drawdown * 100) / ideal_drawdown
            else:
                drawdown_penalty = (max_drawdown * 100 - ideal_drawdown) / 2
        
        # Bonus pour trade rate proche de l'idéal
        if trades_per_month < 20:  # Trop peu de trades
            trade_bonus = -10  # Pénalité forte
        elif trades_per_month > 300:  # Trop de trades
            trade_bonus = -5  # Pénalité modérée
        else:
            # Bonus optimal pour un nombre de trades proche de l'idéal
            trade_diff_pct = abs(trades_per_month - ideal_trades_per_month) / ideal_trades_per_month
            trade_bonus = 5 * (1 - min(1, trade_diff_pct))
        
        # Bonus pour win rate
        if win_rate < 50:  # Win rate trop faible
            win_rate_bonus = -10
        else:
            win_rate_diff = win_rate - ideal_win_rate
            if win_rate_diff < 0:
                win_rate_bonus = win_rate_diff / 5  # Pénalité légère pour win rate inférieur
            else:
                win_rate_bonus = win_rate_diff / 10  # Bonus pour win rate supérieur
        
        # Bonus pour sharpe ratio
        sharpe_bonus = sharpe_ratio * 5
        
        # Calculer distance au rendement mensuel cible
        monthly_return_diff = monthly_return - ideal_monthly_return
        
        # Fonction de fitness principale:
        # - Maximiser pour les rendements proches du cible ou supérieurs
        # - Minimiser les drawdowns et pénaliser les rendements trop faibles
        if monthly_return < 5:  # Rendement mensuel inférieur à 5%
            # Pénalité forte pour les rendements trop faibles
            return monthly_return - drawdown_penalty + trade_bonus/2 + win_rate_bonus/2 + sharpe_bonus/2
        elif monthly_return < ideal_monthly_return:
            # Entre 5% et la cible, pénalité modérée
            return monthly_return - drawdown_penalty + trade_bonus + win_rate_bonus + sharpe_bonus
        else:
            # Au-dessus de la cible, bonus qui augmente moins rapidement
            excess_return = (monthly_return - ideal_monthly_return) / 2
            return ideal_monthly_return + excess_return - drawdown_penalty + trade_bonus + win_rate_bonus + sharpe_bonus
    
    def _generate_random_params(self, strategy_name, iteration=0, best_params=None):
        """
        Génère un ensemble aléatoire de paramètres, éventuellement autour des meilleurs paramètres précédents
        
        Args:
            strategy_name (str): Nom de la stratégie
            iteration (int): Numéro d'itération actuel
            best_params (dict, optional): Meilleurs paramètres trouvés précédemment
            
        Returns:
            dict: Paramètres générés
        """
        param_space = self.param_spaces[strategy_name]
        params = {}
        
        # Si nous avons des meilleurs paramètres et que ce n'est pas la première itération,
        # on les utilise comme point de départ (avec une certaine variance)
        exploration_rate = max(0.1, 1.0 - (iteration / self.max_iterations))
        
        # Technique d'exploration améliorée: plus explorative au début, plus exploitative à la fin
        if iteration < self.max_iterations * 0.3:
            # Phase d'exploration: plus de diversité
            exploration_rate = max(0.5, exploration_rate)
        elif iteration > self.max_iterations * 0.7:
            # Phase d'exploitation: se concentrer sur le raffinement
            exploration_rate = min(0.2, exploration_rate)
        
        for param, space in param_space.items():
            # Si les meilleurs paramètres existent et que ce n'est pas la première itération
            if best_params and param in best_params and iteration > 0:
                if isinstance(space, list):
                    # Pour les paramètres booléens ou catégoriels
                    if random.random() < exploration_rate:
                        params[param] = random.choice(space)
                    else:
                        params[param] = best_params[param]
                elif isinstance(space, tuple) and len(space) == 2:
                    # Pour les paramètres numériques, ajouter une variation
                    min_val, max_val = space
                    current_val = best_params[param]
                    
                    # Variance diminue avec l'itération
                    variance = (max_val - min_val) * exploration_rate
                    
                    # Générer une nouvelle valeur autour de la meilleure valeur
                    if isinstance(min_val, int):
                        # Paramètres entiers
                        new_val = int(current_val + random.uniform(-variance, variance))
                        params[param] = max(min_val, min(max_val, new_val))
                    else:
                        # Paramètres flottants
                        new_val = current_val + random.uniform(-variance, variance)
                        params[param] = max(min_val, min(max_val, new_val))
                else:
                    # Cas par défaut
                    params[param] = best_params[param]
            else:
                # Première itération ou paramètre non présent dans les meilleurs
                if isinstance(space, list):
                    params[param] = random.choice(space)
                elif isinstance(space, tuple) and len(space) == 2:
                    min_val, max_val = space
                    if isinstance(min_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                else:
                    # Valeur par défaut
                    params[param] = space
        
        # Ajouter mutation occasionnelle pour sortir des optima locaux
        if random.random() < 0.1:  # 10% de chance de mutation
            param_to_mutate = random.choice(list(param_space.keys()))
            space = param_space[param_to_mutate]
            
            if isinstance(space, list):
                params[param_to_mutate] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                min_val, max_val = space
                if isinstance(min_val, int):
                    params[param_to_mutate] = random.randint(min_val, max_val)
                else:
                    params[param_to_mutate] = random.uniform(min_val, max_val)
        
        return params
    
    def _initialize_strategy(self, strategy_name, params):
        """
        Initialise une stratégie avec les paramètres donnés
        
        Args:
            strategy_name (str): Nom de la stratégie
            params (dict): Paramètres de la stratégie
            
        Returns:
            BaseStrategy: Instance de la stratégie
        """
        if strategy_name == "trend":
            return TrendFollowingStrategy(params)
        elif strategy_name == "mean_reversion":
            return MeanReversionStrategy(params)
        else:  # combined
            return CombinedStrategy(params)
    
    def _run_backtest(self, strategy, params):
        """
        Exécute un backtest avec la stratégie et les paramètres donnés
        
        Args:
            strategy: Instance de la stratégie
            params (dict): Paramètres utilisés
            
        Returns:
            dict: Résultats du backtest
        """
        try:
            # Récupérer les données historiques
            data = self.fetcher.fetch_historical_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if data is None or data.empty:
                logger.error(f"Pas de données pour {self.symbol} sur {self.timeframe}")
                return None
            
            # Ajouter les indicateurs techniques
            data = TechnicalIndicators.add_all_indicators(data)
            
            # Générer les signaux
            signals = strategy.run(data)
            
            if signals is None or signals.empty or (signals != 0).sum() == 0:
                logger.warning(f"Pas de signaux générés pour {self.symbol} avec les paramètres: {params}")
                return {
                    'params': params,
                    'profit_loss_pct': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'monthly_return': 0.0,
                    'fitness': -100,  # Pénalité forte pour absence de signaux
                    'strategy_name': strategy.name
                }
            
            # Initialiser le trader
            trader = Trader(position_sizer=self.position_sizer, mode="backtest")
            
            # Exécuter le backtest
            equity_history = [CAPITAL.get('initial_capital', 10000)]
            dates = [data.index[0]]
            
            # Exécuter les trades
            for i in range(1, len(data)):
                # Date actuelle
                current_date = data.index[i]
                
                # Données jusqu'à l'index actuel
                current_data = data.iloc[:i+1]
                
                # Signal actuel
                current_signal = signals.iloc[i]
                
                # Prix actuel
                current_price = data['close'].iloc[i]
                
                # Exécuter le signal
                if current_signal != 0:
                    trader.execute_signal(self.symbol, current_signal, current_data)
                
                # Mettre à jour les trailing stops
                for sym in list(trader.active_positions.keys()):
                    trader.update_trailing_stops(sym, current_price)
                
                # Mettre à jour l'équité
                equity_history.append(trader.current_capital)
                dates.append(current_date)
            
            # Récupérer les statistiques
            stats = trader.get_stats()
            
            # Calculer des métriques supplémentaires
            equity_curve = pd.DataFrame({
                'date': dates,
                'equity': equity_history
            }).set_index('date')
            
            # Calcul du ratio de Sharpe
            if len(equity_curve) > 1:
                equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
                sharpe_ratio = equity_curve['returns'].mean() / equity_curve['returns'].std() * np.sqrt(252) if equity_curve['returns'].std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calcul du drawdown
            if len(equity_curve) > 0:
                equity_curve['peak'] = equity_curve['equity'].cummax()
                equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak']
                max_drawdown = equity_curve['drawdown'].max()
            else:
                max_drawdown = 0
            
            # Profitabilité
            profit_loss_pct = stats.get('profit_loss_pct', 0)
            
            # Calculer le rendement mensuel
            if self.test_duration_months > 0:
                monthly_return = profit_loss_pct / self.test_duration_months
            else:
                monthly_return = profit_loss_pct
            
            # Calculer le score de fitness
            fitness = self._calculate_fitness({
                'profit_loss_pct': profit_loss_pct,
                'max_drawdown': max_drawdown,
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats.get('total_trades', 0),
                'sharpe_ratio': sharpe_ratio
            })
            
            # Résultat complet
            result = {
                'params': params,
                'profit_loss_pct': profit_loss_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats.get('total_trades', 0),
                'monthly_return': monthly_return,
                'fitness': fitness,
                'strategy_name': strategy.name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest avec les paramètres {params}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _update_and_apply_best_params(self, best_params, best_result):
        """
        Met à jour et applique immédiatement les meilleurs paramètres trouvés
        
        Args:
            best_params (dict): Meilleurs paramètres
            best_result (dict): Meilleur résultat
        """
        # Mettre à jour les fichiers config pour appliquer les paramètres
        if self.strategy_name == "trend":
            config_var = "TREND_FOLLOWING_PARAMS"
        elif self.strategy_name == "mean_reversion":
            config_var = "MEAN_REVERSION_PARAMS" 
        else:  # combined
            config_var = "COMBINED_STRATEGY_PARAMS"
        
        # Chemin vers le fichier de config
        config_file = "config/strategy_params.py"
        
        # Lire le fichier de configuration actuel
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Chercher la section des paramètres pour cette stratégie
        import re
        pattern = f"{config_var} = {{[^}}]*}}"
        
        # Formater les nouveaux paramètres
        params_str = f"{config_var} = {{\n"
        for k, v in best_params.items():
            if isinstance(v, bool):
                params_str += f"    \"{k}\": {str(v)},\n"
            elif isinstance(v, (int, float)):
                params_str += f"    \"{k}\": {v},\n"
            else:
                params_str += f"    \"{k}\": \"{v}\",\n"
        params_str += "}"
        
        # Remplacer les paramètres dans le fichier
        if re.search(pattern, config_content):
            new_config = re.sub(pattern, params_str, config_content)
        else:
            # Si la section n'existe pas, l'ajouter à la fin
            new_config = config_content + "\n\n" + params_str + "\n"
        
        # Écrire le nouveau fichier de configuration
        with open(config_file, 'w') as f:
            f.write(new_config)
        
        logger.info(f"Paramètres mis à jour dans {config_file}")
        
        # Afficher un rapport sur les nouveaux paramètres
        print("\n===== NOUVEAUX PARAMÈTRES APPLIQUÉS =====")
        print(f"Stratégie: {self.strategy_name}")
        print(f"Rendement total: {best_result['profit_loss_pct']:.2f}%")
        print(f"Rendement mensuel: {best_result['monthly_return']:.2f}%")
        print(f"Score de fitness: {best_result['fitness']:.2f}")
        print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {best_result['win_rate']:.2f}%")
        print(f"Total Trades: {best_result['total_trades']} ({best_result['total_trades']/self.test_duration_months:.1f}/mois)")
        print("\nParamètres:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    def _generate_interim_report(self, iteration, results, best_result):
        """
        Génère un rapport intermédiaire pour l'itération actuelle
        
        Args:
            iteration (int): Numéro d'itération
            results (list): Résultats de l'itération
            best_result (dict): Meilleur résultat global
        """
        report_file = self.results_dir / f"rapport_iteration_{iteration+1}_{self.strategy_name}_{self.timeframe}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"====== RAPPORT D'OPTIMISATION CRYPTO - ITÉRATION {iteration+1}/{self.max_iterations} ======\n\n")
            f.write(f"Stratégie: {self.strategy_name}\n")
            f.write(f"Symbole: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Période: {self.start_date.strftime('%Y-%m-%d')} à {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Durée du test: {self.test_duration_months:.2f} mois\n")
            f.write(f"Rendement mensuel cible: {self.target_monthly_return:.2f}%\n\n")
            
            if self.stagnation_counter > 0:
                f.write(f"Stagnation: {self.stagnation_counter}/{self.stagnation_limit} itérations sans amélioration\n\n")
            
            f.write("--- RÉSULTATS DE L'ITÉRATION ACTUELLE ---\n\n")
            f.write(f"Nombre de configurations testées: {len(results)}\n\n")
            
            if results:
                f.write("Top 5 résultats de cette itération (triés par score de fitness):\n")
                for i, result in enumerate(results[:5]):
                    f.write(f"#{i+1}: Rendement mensuel={result['monthly_return']:.2f}%, ")
                    f.write(f"Rendement total={result['profit_loss_pct']:.2f}%, ")
                    f.write(f"Fitness={result['fitness']:.2f}, ")
                    f.write(f"Sharpe={result['sharpe_ratio']:.2f}, ")
                    f.write(f"Drawdown={result['max_drawdown']*100:.2f