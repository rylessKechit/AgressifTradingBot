"""
Script d'auto-optimisation du bot de trading
Optimise automatiquement les paramètres jusqu'à atteindre un rendement cible
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

class AutoOptimizer:
    """
    Classe pour l'optimisation automatique des stratégies de trading
    """
    
    def __init__(self, target_return=15.0, max_iterations=20, population_size=20, 
                 symbol="BTC/USDT", timeframe="15m", start_date=None, end_date=None,
                 strategy_name="combined", results_dir="optimization_results"):
        """
        Initialise l'optimiseur automatique
        
        Args:
            target_return (float): Rendement cible en pourcentage
            max_iterations (int): Nombre maximum d'itérations
            population_size (int): Taille de la population par itération
            symbol (str): Symbole de la paire à trader
            timeframe (str): Timeframe
            start_date (str): Date de début (format YYYY-MM-DD)
            end_date (str): Date de fin (format YYYY-MM-DD)
            strategy_name (str): Nom de la stratégie ("trend", "mean_reversion", "combined")
            results_dir (str): Répertoire pour sauvegarder les résultats
        """
        self.target_return = target_return
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Convertir les dates si fournies
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = datetime.now() - timedelta(days=180)  # Par défaut: 6 mois
            
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
        self.position_sizer = PositionSizer(
            initial_capital=CAPITAL.get('initial_capital', 10000),
            max_risk_per_trade=CAPITAL.get('risk_per_trade', 0.01),
            max_position_size=CAPITAL.get('max_position_size', 0.2)
        )
        
        # Espaces de paramètres pour chaque stratégie
        self.param_spaces = self._init_param_spaces()
        
        logger.info(f"Auto-optimiseur initialisé avec cible de rendement de {target_return}%")
        logger.info(f"Stratégie: {strategy_name}, Paire: {symbol}, Timeframe: {timeframe}")
        logger.info(f"Période: {self.start_date.strftime('%Y-%m-%d')} à {self.end_date.strftime('%Y-%m-%d')}")
        
    def _init_param_spaces(self):
        """
        Initialise les espaces de paramètres pour chaque stratégie
        
        Returns:
            dict: Espaces de paramètres
        """
        param_spaces = {
            "trend": {
                # Périodes des moyennes mobiles
                "fast_ema": (5, 15),          # Plage: 5-15
                "medium_ema": (15, 35),       # Plage: 15-35
                "slow_ema": (35, 70),         # Plage: 35-70
                "very_slow_ema": (100, 250),  # Plage: 100-250
                
                # MACD
                "macd_fast": (8, 16),         # Plage: 8-16
                "macd_slow": (20, 30),        # Plage: 20-30
                "macd_signal": (7, 12),       # Plage: 7-12
                
                # ADX
                "adx_threshold": (15, 35),    # Plage: 15-35
                
                # Seuils de signaux
                "buy_threshold": (0.3, 0.9),  # Plage: 0.3-0.9
                "sell_threshold": (-0.9, -0.3), # Plage: -0.9 à -0.3
                
                # Poids
                "weight": (0.3, 0.6)          # Plage: 0.3-0.6
            },
            "mean_reversion": {
                # RSI
                "rsi_period": (7, 21),        # Plage: 7-21
                "rsi_overbought": (65, 80),   # Plage: 65-80
                "rsi_oversold": (20, 35),     # Plage: 20-35
                
                # Bandes de Bollinger
                "bb_period": (15, 30),        # Plage: 15-30
                "bb_std": (1.5, 3.0),         # Plage: 1.5-3.0
                
                # Stochastique
                "stoch_k": (10, 20),          # Plage: 10-20
                "stoch_d": (3, 7),            # Plage: 3-7
                "stoch_overbought": (70, 85), # Plage: 70-85
                "stoch_oversold": (15, 30),   # Plage: 15-30
                
                # Seuils de signaux
                "buy_threshold": (0.3, 0.9),  # Plage: 0.3-0.9
                "sell_threshold": (-0.9, -0.3), # Plage: -0.9 à -0.3
                
                # Poids
                "weight": (0.3, 0.6)          # Plage: 0.3-0.6
            },
            "combined": {
                # Seuils finaux
                "final_buy_threshold": (0.2, 0.7),   # Plage: 0.2-0.7
                "final_sell_threshold": (-0.7, -0.2), # Plage: -0.7 à -0.2
                
                # Filtres
                "adjust_weights_by_volatility": [True, False],
                "use_volume_filter": [True, False],
                "volume_threshold": (1.0, 3.0),      # Plage: 1.0-3.0
                "use_multi_timeframe": [True, False],
                "cooldown_period": (1, 10),          # Plage: 1-10
                
                # Timeframe
                "timeframe_alignment_required": [True, False],
            }
        }
        
        return param_spaces
    
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
            
            result = {
                'params': params,
                'profit_loss_pct': stats.get('profit_loss_pct', 0),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats.get('total_trades', 0),
                'strategy_name': strategy.name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest avec les paramètres {params}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_optimization(self):
        """
        Exécute le processus d'optimisation
        
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
                    logger.info(f"Résultat: PnL={result['profit_loss_pct']:.2f}%, Sharpe={result['sharpe_ratio']:.2f}, DD={result['max_drawdown']*100:.2f}%, Trades={result['total_trades']}")
            
            # Si aucun résultat valide, passer à l'itération suivante
            if not results:
                logger.warning(f"Aucun résultat valide à l'itération {iteration+1}, continuez...")
                continue
                
            # Trier les résultats par rendement
            results.sort(key=lambda x: (x['profit_loss_pct'], x['sharpe_ratio']), reverse=True)
            
            # Mettre à jour le meilleur résultat si nécessaire
            if not best_result or results[0]['profit_loss_pct'] > best_result['profit_loss_pct']:
                best_result = results[0]
                best_params = results[0]['params']
                logger.info(f"Nouveau meilleur résultat: {best_result['profit_loss_pct']:.2f}%")
            
            # Sauvegarder les résultats de l'itération
            self.optimization_history.append({
                'iteration': iteration + 1,
                'results': results,
                'best_result': best_result.copy() if best_result else None
            })
            
            # Sauvegarder l'historique
            self._save_optimization_history()
            
            # Vérifier si l'objectif est atteint
            if best_result and best_result['profit_loss_pct'] >= self.target_return:
                logger.info(f"Objectif atteint à l'itération {iteration+1}: {best_result['profit_loss_pct']:.2f}% >= {self.target_return}%")
                break
                
            # Afficher un résumé de l'itération
            self._print_iteration_summary(iteration, results, best_result)
        
        # Afficher et sauvegarder les résultats finaux
        if best_result:
            logger.info("\n=== MEILLEUR RÉSULTAT TROUVÉ ===")
            logger.info(f"Rendement: {best_result['profit_loss_pct']:.2f}%")
            logger.info(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
            logger.info(f"Win Rate: {best_result['win_rate']:.2f}%")
            logger.info(f"Total Trades: {best_result['total_trades']}")
            logger.info("\nParamètres:")
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")
            
            # Sauvegarder les meilleurs paramètres
            self._save_best_params(best_params, best_result)
            
            # Tracer les performances
            self._plot_optimization_performance()
        else:
            logger.warning("Aucun résultat valide trouvé après optimisation")
        
        return best_params
    
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
            print("\nTop 3 résultats de cette itération:")
            for i, result in enumerate(results[:3]):
                print(f"#{i+1}: PnL={result['profit_loss_pct']:.2f}%, Sharpe={result['sharpe_ratio']:.2f}, DD={result['max_drawdown']*100:.2f}%, Trades={result['total_trades']}")
            
            print("\nParamètres du meilleur résultat de cette itération:")
            for param, value in results[0]['params'].items():
                print(f"  {param}: {value}")
        
        if best_result:
            print(f"\nMeilleur résultat global: PnL={best_result['profit_loss_pct']:.2f}%, Sharpe={best_result['sharpe_ratio']:.2f}")
    
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
                'optimization_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(param_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Meilleurs paramètres sauvegardés dans {param_file}")
    
    def _plot_optimization_performance(self):
        """
        Trace un graphique montrant l'évolution des performances pendant l'optimisation
        """
        if not self.optimization_history:
            return
            
        iterations = [item['iteration'] for item in self.optimization_history]
        
        # Extraire les mesures de performance
        best_returns = []
        avg_returns = []
        best_sharpes = []
        avg_sharpes = []
        
        for item in self.optimization_history:
            if item['best_result']:
                best_returns.append(item['best_result']['profit_loss_pct'])
                best_sharpes.append(item['best_result']['sharpe_ratio'])
            else:
                best_returns.append(0)
                best_sharpes.append(0)
                
            if item['results']:
                avg_returns.append(np.mean([r['profit_loss_pct'] for r in item['results']]))
                avg_sharpes.append(np.mean([r['sharpe_ratio'] for r in item['results']]))
            else:
                avg_returns.append(0)
                avg_sharpes.append(0)
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Rendements
        plt.subplot(2, 1, 1)
        plt.plot(iterations, best_returns, 'b-', marker='o', label='Meilleur rendement')
        plt.plot(iterations, avg_returns, 'g--', marker='x', label='Rendement moyen')
        plt.axhline(y=self.target_return, color='r', linestyle='-', label=f'Cible ({self.target_return}%)')
        plt.title('Évolution des rendements pendant l\'optimisation')
        plt.xlabel('Itération')
        plt.ylabel('Rendement (%)')
        plt.grid(True)
        plt.legend()
        
        # Sharpe Ratio
        plt.subplot(2, 1, 2)
        plt.plot(iterations, best_sharpes, 'b-', marker='o', label='Meilleur Sharpe')
        plt.plot(iterations, avg_sharpes, 'g--', marker='x', label='Sharpe moyen')
        plt.title('Évolution du ratio de Sharpe pendant l\'optimisation')
        plt.xlabel('Itération')
        plt.ylabel('Ratio de Sharpe')
        plt.grid(True)
        plt.legend()
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        plot_file = self.results_dir / f"optimization_performance_{self.strategy_name}_{self.timeframe}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_file)
        
        logger.info(f"Graphique de performance sauvegardé dans {plot_file}")
        
        # Fermer la figure
        plt.close()


def main():
    """
    Fonction principale
    """
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Auto-optimisation du bot de trading")
    
    parser.add_argument('--target', type=float, default=15.0,
                       help='Rendement cible en pourcentage')
    
    parser.add_argument('--iterations', type=int, default=20,
                       help='Nombre maximum d\'itérations')
    
    parser.add_argument('--population', type=int, default=20,
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
    
    args = parser.parse_args()
    
    # Initialiser l'optimiseur
    optimizer = AutoOptimizer(
        target_return=args.target,
        max_iterations=args.iterations,
        population_size=args.population,
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end
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