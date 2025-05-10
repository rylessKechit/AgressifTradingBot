"""
Script d'optimisation agressive du bot de trading
Optimise pour un rendement mensuel maximal avec tolérance au risque élevée
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

class AggressiveOptimizer:
    """
    Classe pour l'optimisation agressive des stratégies de trading
    """
    
    def __init__(self, target_monthly_return=5.0, max_iterations=30, population_size=25, 
                 symbol="BTC/USDT", timeframe="15m", start_date=None, end_date=None,
                 strategy_name="combined", results_dir="aggressive_optimization_results",
                 max_acceptable_drawdown=30.0):
        """
        Initialise l'optimiseur agressif
        
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
        """
        self.target_monthly_return = target_monthly_return
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_acceptable_drawdown = max_acceptable_drawdown
        
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
        
        # Initialiser les composants
        self.fetcher = DataFetcher()
        
        # Créer un position sizer plus agressif
        self.position_sizer = PositionSizer(
            initial_capital=CAPITAL.get('initial_capital', 10000),
            max_risk_per_trade=0.03,  # Augmenté à 3% par trade
            max_position_size=0.4  # Augmenté à 40% du capital
        )
        
        # Espaces de paramètres pour chaque stratégie (plages plus larges)
        self.param_spaces = self._init_aggressive_param_spaces()
        
        # Calculer la durée du test en mois pour normaliser le rendement
        delta = self.end_date - self.start_date
        self.test_duration_months = delta.days / 30.44  # Approximation du nombre de mois
        
        # Calculer le rendement cible total pour la période
        self.target_total_return = (1 + target_monthly_return/100) ** self.test_duration_months * 100 - 100
        
        logger.info(f"Optimiseur agressif initialisé avec cible de rendement mensuel de {target_monthly_return}%")
        logger.info(f"Pour la période de test, cela correspond à un rendement total cible de {self.target_total_return:.2f}%")
        logger.info(f"Stratégie: {strategy_name}, Paire: {symbol}, Timeframe: {timeframe}")
        logger.info(f"Période: {self.start_date.strftime('%Y-%m-%d')} à {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Drawdown maximum acceptable: {max_acceptable_drawdown}%")
        
    def _init_aggressive_param_spaces(self):
        """
        Initialise les espaces de paramètres agressifs pour chaque stratégie
        
        Returns:
            dict: Espaces de paramètres
        """
        param_spaces = {
            "trend": {
                # Périodes des moyennes mobiles (plages plus étroites pour plus de réactivité)
                "fast_ema": (3, 12),          # Plage réduite pour plus de réactivité
                "medium_ema": (10, 25),       # Plage réduite
                "slow_ema": (30, 60),         # Plage réduite
                "very_slow_ema": (100, 200),  # Maintenu
                
                # MACD (paramètres plus agressifs)
                "macd_fast": (6, 12),         # Plage réduite
                "macd_slow": (18, 26),        # Plage réduite
                "macd_signal": (5, 9),        # Plage réduite
                
                # ADX
                "adx_threshold": (15, 25),    # Seuil plus bas pour entrer plus tôt dans les tendances
                
                # Seuils de signaux (plus sensibles)
                "buy_threshold": (0.2, 0.7),  # Seuil d'achat plus bas
                "sell_threshold": (-0.7, -0.2), # Seuil de vente plus haut
                
                # Poids
                "weight": (0.3, 0.7)          # Plage élargie
            },
            "mean_reversion": {
                # RSI (plus sensible aux surachats/surventes)
                "rsi_period": (5, 14),        # Périodes plus courtes
                "rsi_overbought": (65, 75),   # Seuil plus bas pour entrer plus tôt
                "rsi_oversold": (25, 35),     # Seuil plus haut pour entrer plus tôt
                
                # Bandes de Bollinger
                "bb_period": (10, 20),        # Périodes plus courtes
                "bb_std": (1.5, 2.5),         # Moins d'écarts types pour des bandes plus serrées
                
                # Stochastique (plus sensible)
                "stoch_k": (8, 14),           # Périodes plus courtes
                "stoch_d": (2, 5),            # Périodes plus courtes
                "stoch_overbought": (70, 80), # Plage standard
                "stoch_oversold": (20, 30),   # Plage standard
                
                # Seuils de signaux (plus sensibles)
                "buy_threshold": (0.2, 0.7),  # Seuil d'achat plus bas
                "sell_threshold": (-0.7, -0.2), # Seuil de vente plus haut
                
                # Poids
                "weight": (0.3, 0.7)          # Plage élargie
            },
            "combined": {
                # Seuils finaux (plus sensibles)
                "final_buy_threshold": (0.15, 0.5),   # Seuil plus bas
                "final_sell_threshold": (-0.5, -0.15), # Seuil plus haut
                
                # Filtres (moins restrictifs)
                "adjust_weights_by_volatility": [True, False],
                "use_volume_filter": [True, False],
                "volume_threshold": (1.0, 2.5),       # Seuil plus bas
                "use_multi_timeframe": [True, False],
                "cooldown_period": (1, 5),            # Période plus courte
                
                # Timeframe
                "timeframe_alignment_required": [False],  # Désactivé par défaut pour plus de signaux
            }
        }
        
        return param_spaces
    
    def _calculate_fitness(self, result):
        """
        Calcule le score de fitness d'un résultat
        Priorise le rendement total avec une pénalité limitée pour le drawdown
        
        Args:
            result (dict): Résultat du backtest
            
        Returns:
            float: Score de fitness
        """
        profit_loss_pct = result.get('profit_loss_pct', 0)
        max_drawdown = result.get('max_drawdown', 1)
        total_trades = result.get('total_trades', 0)
        
        # Drawdown pénalité (seulement si > max_acceptable_drawdown)
        if max_drawdown * 100 > self.max_acceptable_drawdown:
            drawdown_penalty = (max_drawdown * 100 - self.max_acceptable_drawdown) * 0.5
        else:
            drawdown_penalty = 0
            
        # Bonus pour le nombre de trades (pour éviter les stratégies inactives)
        trade_bonus = min(total_trades / 100, 5) if total_trades > 0 else 0
        
        # Calculer le rendement mensuel (annualisé puis divisé par 12)
        if self.test_duration_months > 0:
            monthly_return = profit_loss_pct / self.test_duration_months
        else:
            monthly_return = profit_loss_pct
        
        # Formule de fitness: rendement mensuel est primordial
        fitness = monthly_return - drawdown_penalty + trade_bonus
        
        return fitness
    
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
                'total_trades': stats.get('total_trades', 0)
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
        print(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {best_result['win_rate']:.2f}%")
        print(f"Total Trades: {best_result['total_trades']}")
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
            f.write(f"====== RAPPORT D'OPTIMISATION AGRESSIVE - ITÉRATION {iteration+1}/{self.max_iterations} ======\n\n")
            f.write(f"Stratégie: {self.strategy_name}\n")
            f.write(f"Symbole: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Période: {self.start_date.strftime('%Y-%m-%d')} à {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Durée du test: {self.test_duration_months:.2f} mois\n")
            f.write(f"Rendement mensuel cible: {self.target_monthly_return:.2f}%\n\n")
            
            f.write("--- RÉSULTATS DE L'ITÉRATION ACTUELLE ---\n\n")
            f.write(f"Nombre de configurations testées: {len(results)}\n\n")
            
            if results:
                f.write("Top 5 résultats de cette itération (triés par score de fitness):\n")
                for i, result in enumerate(results[:5]):
                    f.write(f"#{i+1}: Rendement mensuel={result['monthly_return']:.2f}%, ")
                    f.write(f"Rendement total={result['profit_loss_pct']:.2f}%, ")
                    f.write(f"Fitness={result['fitness']:.2f}, ")
                    f.write(f"Drawdown={result['max_drawdown']*100:.2f}%, ")
                    f.write(f"Trades={result['total_trades']}, ")
                    f.write(f"Win Rate={result['win_rate']:.2f}%\n")
                
                f.write("\nParamètres du meilleur résultat de cette itération:\n")
                for param, value in results[0]['params'].items():
                    f.write(f"  {param}: {value}\n")
            
            f.write("\n--- MEILLEUR RÉSULTAT GLOBAL ---\n\n")
            if best_result:
                f.write(f"Rendement mensuel: {best_result['monthly_return']:.2f}%\n")
                f.write(f"Rendement total: {best_result['profit_loss_pct']:.2f}%\n")
                f.write(f"Score de fitness: {best_result['fitness']:.2f}\n")
                f.write(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%\n")
                f.write(f"Win Rate: {best_result['win_rate']:.2f}%\n")
                f.write(f"Total Trades: {best_result['total_trades']}\n\n")
                
                f.write("Paramètres:\n")
                for param, value in best_result['params'].items():
                    f.write(f"  {param}: {value}\n")
            else:
                f.write("Aucun résultat valide trouvé.\n")
        
        logger.info(f"Rapport intermédiaire sauvegardé dans {report_file}")
    
    def _print_iteration_summary(self, iteration, results, best_result):
        """
        Affiche un résumé de l'itération
        
        Args:
            iteration (int): Numéro d'itération
            results (list): Résultats de l'itération
            best_result (dict): Meilleur résultat global
        """
        print(f"\n--- Résumé de l'itération {iteration+1} ---")
        print(f"Nombre de résultats valides: {len(results)}")
        
        if results:
            print("\nTop 3 résultats de cette itération (triés par fitness):")
            for i, result in enumerate(results[:3]):
                print(f"#{i+1}: Rendement mensuel={result['monthly_return']:.2f}%, Total={result['profit_loss_pct']:.2f}%, ",
                     f"Fitness={result['fitness']:.2f}, DD={result['max_drawdown']*100:.2f}%, Trades={result['total_trades']}")
            
            print("\nParamètres du meilleur résultat de cette itération:")
            for param, value in results[0]['params'].items():
                print(f"  {param}: {value}")
        
        if best_result:
            print(f"\nMeilleur résultat global: ")
            print(f"  Rendement mensuel: {best_result['monthly_return']:.2f}%")
            print(f"  Rendement total: {best_result['profit_loss_pct']:.2f}%") 
            print(f"  Fitness: {best_result['fitness']:.2f}")
            print(f"  Drawdown: {best_result['max_drawdown']*100:.2f}%")
    
    def _save_optimization_history(self):
        """
        Sauvegarde l'historique d'optimisation dans un fichier
        """
        history_file = self.results_dir / f"optimization_history_{self.strategy_name}_{self.timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Convertir en format serializable
        serializable_history = []
        for item in self.optimization_history:
            serializable_item = {
                'iteration': item['iteration'],
                'results': [],
                'best_result': None
            }
            
            for result in item['results']:
                serializable_result = {k: (float(v) if isinstance(v, np.float64) else v) 
                                      for k, v in result.items() if k != 'params'}
                serializable_result['params'] = {k: (float(v) if isinstance(v, np.float64) else v) 
                                               for k, v in result['params'].items()}
                serializable_item['results'].append(serializable_result)
            
            if item['best_result']:
                best_result = {k: (float(v) if isinstance(v, np.float64) else v) 
                              for k, v in item['best_result'].items() if k != 'params'}
                best_result['params'] = {k: (float(v) if isinstance(v, np.float64) else v) 
                                       for k, v in item['best_result']['params'].items()}
                serializable_item['best_result'] = best_result
                
            serializable_history.append(serializable_item)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        logger.info(f"Historique d'optimisation sauvegardé dans {history_file}")
    
    def _save_best_params(self, params, result):
        """
        Sauvegarde les meilleurs paramètres dans un fichier
        
        Args:
            params (dict): Meilleurs paramètres
            result (dict): Résultat associé
        """
        # Créer un dictionnaire avec les paramètres et le résultat
        param_file = self.results_dir / f"best_params_{self.strategy_name}_{self.timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Convertir en format serializable
        serializable_params = {k: (float(v) if isinstance(v, np.float64) else v) for k, v in params.items()}
        serializable_result = {k: (float(v) if isinstance(v, np.float64) else v) 
                              for k, v in result.items() if k != 'params'}
        
        output = {
            'params': serializable_params,
            'result': serializable_result,
            'metadata': {
                'strategy': self.strategy_name,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'optimization_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_duration_months': float(self.test_duration_months),
                'target_monthly_return': float(self.target_monthly_return)
            }
        }
        
        with open(param_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Meilleurs paramètres sauvegardés dans {param_file}")
    
    def _plot_optimization_performance(self, interim=False):
        """
        Trace un graphique montrant l'évolution des performances pendant l'optimisation
        
        Args:
            interim (bool): Si True, c'est un graphique intermédiaire
        """
        if not self.optimization_history:
            return
            
        iterations = [item['iteration'] for item in self.optimization_history]
        
        # Extraire les mesures de performance
        best_monthly_returns = []
        avg_monthly_returns = []
        best_drawdowns = []
        best_fitness = []
        total_trades = []
        
        for item in self.optimization_history:
            if item['best_result']:
                best_monthly_returns.append(item['best_result'].get('monthly_return', 0))
                best_drawdowns.append(item['best_result'].get('max_drawdown', 0) * 100)
                best_fitness.append(item['best_result'].get('fitness', 0))
                total_trades.append(item['best_result'].get('total_trades', 0))
            else:
                best_monthly_returns.append(0)
                best_drawdowns.append(0)
                best_fitness.append(0)
                total_trades.append(0)
                
            if item['results']:
                avg_monthly_returns.append(np.mean([r.get('monthly_return', 0) for r in item['results']]))
            else:
                avg_monthly_returns.append(0)
        
        # Créer la figure
        plt.figure(figsize=(15, 12))
        
        # Rendements mensuels
        plt.subplot(3, 1, 1)
        plt.plot(iterations, best_monthly_returns, 'b-', marker='o', label='Meilleur rendement mensuel')
        plt.plot(iterations, avg_monthly_returns, 'g--', marker='x', label='Rendement mensuel moyen')
        plt.axhline(y=self.target_monthly_return, color='r', linestyle='-', label=f'Cible ({self.target_monthly_return}%)')
        plt.title('Évolution des rendements mensuels pendant l\'optimisation')
        plt.xlabel('Itération')
        plt.ylabel('Rendement mensuel (%)')
        plt.grid(True)
        plt.legend()
        
        # Drawdown
        plt.subplot(3, 1, 2)
        plt.plot(iterations, best_drawdowns, 'r-', marker='o')
        plt.axhline(y=self.max_acceptable_drawdown, color='r', linestyle='--', label=f'Maximum acceptable ({self.max_acceptable_drawdown}%)')
        plt.title('Évolution du drawdown maximum')
        plt.xlabel('Itération')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        # Score de fitness
        plt.subplot(3, 1, 3)
        plt.plot(iterations, best_fitness, 'b-', marker='o', label='Meilleur fitness')
        plt.title('Évolution du score de fitness')
        plt.xlabel('Itération')
        plt.ylabel('Score de fitness')
        plt.grid(True)
        
        # Ajouter les trades comme ligne secondaire
        ax2 = plt.twinx()
        ax2.plot(iterations, total_trades, 'g--', marker='x', label='Nombre de trades')
        ax2.set_ylabel('Nombre de trades', color='g')
        
        # Légendes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        suffix = "interim" if interim else "final"
        plot_file = self.results_dir / f"optimization_performance_{self.strategy_name}_{self.timeframe}_{suffix}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_file)
        
        logger.info(f"Graphique de performance sauvegardé dans {plot_file}")
        
        # Fermer la figure
        plt.close()
    
    def run_optimization(self):
        """
        Exécute le processus d'optimisation agressive
        
        Returns:
            dict: Meilleurs paramètres trouvés
        """
        best_params = None
        best_result = None
        
        for iteration in range(self.max_iterations):
            logger.info(f"Itération {iteration+1}/{self.max_iterations}")
            
            # Générer la population de paramètres
            population = []
            for _ in range(self.population_size):
                params = self._generate_random_params(self.strategy_name, iteration, best_params)
                population.append(params)
            
            # Exécuter les backtests pour chaque ensemble de paramètres
            results = []
            for i, params in enumerate(population):
                logger.info(f"Backtest {i+1}/{len(population)} avec paramètres: {params}")
                
                # Initialiser la stratégie avec les paramètres
                strategy = self._initialize_strategy(self.strategy_name, params)
                
                # Exécuter le backtest
                result = self._run_backtest(strategy, params)
                
                if result:
                    results.append(result)
                    logger.info(f"Résultat: Rendement mensuel={result['monthly_return']:.2f}%, "
                               f"Total={result['profit_loss_pct']:.2f}%, Fitness={result['fitness']:.2f}, "
                               f"DD={result['max_drawdown']*100:.2f}%, Trades={result['total_trades']}")
            
            # Si aucun résultat valide, passer à l'itération suivante
            if not results:
                logger.warning(f"Aucun résultat valide à l'itération {iteration+1}, continuons...")
                continue
                
            # Trier les résultats par fitness
            results.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Mettre à jour le meilleur résultat si nécessaire
            iteration_best = results[0]
            if not best_result or iteration_best['fitness'] > best_result['fitness']:
                best_result = iteration_best
                best_params = iteration_best['params']
                logger.info(f"Nouveau meilleur résultat: Fitness={best_result['fitness']:.2f}, "
                           f"Rendement mensuel={best_result['monthly_return']:.2f}%")
                
                # Appliquer immédiatement les meilleurs paramètres
                self._update_and_apply_best_params(best_params, best_result)
            
            # Sauvegarder les résultats de l'itération
            self.optimization_history.append({
                'iteration': iteration + 1,
                'results': results,
                'best_result': best_result.copy() if best_result else None
            })
            
            # Sauvegarder l'historique
            self._save_optimization_history()
            
            # Générer et sauvegarder un rapport intermédiaire
            self._generate_interim_report(iteration, results, best_result)
            
            # Vérifier si l'objectif est atteint
            if best_result and best_result['monthly_return'] >= self.target_monthly_return:
                logger.info(f"Objectif atteint à l'itération {iteration+1}: "
                          f"Rendement mensuel {best_result['monthly_return']:.2f}% >= {self.target_monthly_return}%")
                break
                
            # Afficher un résumé de l'itération
            self._print_iteration_summary(iteration, results, best_result)
            
            # Tracer les performances intermédiaires
            self._plot_optimization_performance(interim=True)
        
        # Afficher et sauvegarder les résultats finaux
        if best_result:
            logger.info("\n=== MEILLEUR RÉSULTAT FINAL ===")
            logger.info(f"Rendement mensuel: {best_result['monthly_return']:.2f}%")
            logger.info(f"Rendement total: {best_result['profit_loss_pct']:.2f}%")
            logger.info(f"Score de fitness: {best_result['fitness']:.2f}")
            logger.info(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
            logger.info(f"Win Rate: {best_result['win_rate']:.2f}%")
            logger.info(f"Total Trades: {best_result['total_trades']}")
            logger.info("\nParamètres:")
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")
            
            # Sauvegarder les meilleurs paramètres
            self._save_best_params(best_params, best_result)
            
            # Tracer les performances finales
            self._plot_optimization_performance()
        else:
            logger.warning("Aucun résultat valide trouvé après optimisation")
        
        return best_params


def main():
    """
    Fonction principale
    """
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Optimisation agressive du bot de trading")
    
    parser.add_argument('--target', type=float, default=5.0,
                       help='Rendement mensuel cible en pourcentage')
    
    parser.add_argument('--iterations', type=int, default=30,
                       help='Nombre maximum d\'itérations')
    
    parser.add_argument('--population', type=int, default=25,
                       help='Taille de la population par itération')
    
    parser.add_argument('--strategy', type=str, default='combined',
                       choices=['trend', 'mean_reversion', 'combined'],
                       help='Stratégie à optimiser')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Symbole de la paire')
    
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='Timeframe')
    
    parser.add_argument('--start', type=str, default=None,
                       help='Date de début (format: YYYY-MM-DD)')
    
    parser.add_argument('--end', type=str, default=None,
                       help='Date de fin (format: YYYY-MM-DD)')
    
    parser.add_argument('--max-drawdown', type=float, default=30.0,
                       help='Drawdown maximum acceptable en pourcentage')
    
    args = parser.parse_args()
    
    # Initialiser l'optimiseur
    optimizer = AggressiveOptimizer(
        target_monthly_return=args.target,
        max_iterations=args.iterations,
        population_size=args.population,
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        max_acceptable_drawdown=args.max_drawdown
    )
    
    # Exécuter l'optimisation
    best_params = optimizer.run_optimization()
    
    # Afficher les meilleurs paramètres
    if best_params:
        print("\n=== PARAMÈTRES OPTIMAUX TROUVÉS ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")


if __name__ == "__main__":
    main()