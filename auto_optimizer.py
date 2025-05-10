"""
Script d'auto-optimisation du bot de trading
Optimise automatiquement les paramètres de la stratégie et met à jour la configuration
"""
import os
import sys
import time
import yaml
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules nécessaires
from config.settings import BACKTEST, TRADING, CAPITAL
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.combined_strategy import CombinedStrategy
from backtesting.optimizer import StrategyOptimizer
from backtesting.performance import PerformanceAnalyzer
from utils.logger import setup_logger
from utils.visualizer import plot_backtest_results
from utils.email_notifier import EmailNotifier

# Configuration du logging
logger = setup_logger(level=logging.INFO)

class AutoOptimizer:
    """
    Classe pour l'auto-optimisation du bot de trading
    """
    
    def __init__(self, config_dir='config', results_dir='results'):
        """
        Initialise l'auto-optimiseur
        
        Args:
            config_dir (str, optional): Répertoire des fichiers de configuration
            results_dir (str, optional): Répertoire pour sauvegarder les résultats
        """
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        
        # Créer les répertoires si nécessaires
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Stratégies disponibles
        self.strategies = {
            'trend': TrendFollowingStrategy,
            'mean_reversion': MeanReversionStrategy,
            'combined': CombinedStrategy
        }
        
        # Initialiser le notificateur
        self.notifier = EmailNotifier()
        
        logger.info("Auto-optimiseur initialisé")
    
    def load_strategy_params(self, strategy_name):
        """
        Charge les paramètres actuels de la stratégie depuis le fichier de configuration
        
        Args:
            strategy_name (str): Nom de la stratégie
            
        Returns:
            dict: Paramètres de la stratégie
        """
        # Fichier de paramètres de stratégie
        config_file = self.config_dir / "strategy_params.py"
        
        if not config_file.exists():
            logger.warning(f"Fichier de configuration {config_file} introuvable, utilisation des paramètres par défaut")
            return {}
            
        # Charger le fichier de configuration Python
        # C'est un peu différent d'un fichier YAML/JSON car c'est un module Python
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("strategy_params", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Récupérer les paramètres de la stratégie
            if strategy_name == 'trend':
                return getattr(config_module, 'TREND_FOLLOWING_PARAMS', {})
            elif strategy_name == 'mean_reversion':
                return getattr(config_module, 'MEAN_REVERSION_PARAMS', {})
            elif strategy_name == 'combined':
                return getattr(config_module, 'COMBINED_STRATEGY_PARAMS', {})
            else:
                logger.warning(f"Stratégie {strategy_name} inconnue")
                return {}
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des paramètres de la stratégie: {e}")
            return {}
    
    def save_strategy_params(self, strategy_name, params):
        """
        Sauvegarde les nouveaux paramètres de la stratégie dans le fichier de configuration
        
        Args:
            strategy_name (str): Nom de la stratégie
            params (dict): Nouveaux paramètres de la stratégie
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        # Fichier de paramètres de stratégie
        config_file = self.config_dir / "strategy_params.py"
        
        # Si le fichier n'existe pas, créer un fichier vide
        if not config_file.exists():
            with open(config_file, 'w') as f:
                f.write("# Paramètres des stratégies de trading\n\n")
        
        try:
            # Lire le contenu actuel du fichier
            with open(config_file, 'r') as f:
                content = f.read()
                
            # Nom de la variable pour cette stratégie
            if strategy_name == 'trend':
                var_name = 'TREND_FOLLOWING_PARAMS'
            elif strategy_name == 'mean_reversion':
                var_name = 'MEAN_REVERSION_PARAMS'
            elif strategy_name == 'combined':
                var_name = 'COMBINED_STRATEGY_PARAMS'
            else:
                logger.warning(f"Stratégie {strategy_name} inconnue")
                return False
            
            # Vérifier si la variable existe déjà dans le fichier
            if f"{var_name} = " in content:
                # Mise à jour des paramètres existants
                lines = content.split('\n')
                new_lines = []
                in_var = False
                
                for line in lines:
                    if line.startswith(f"{var_name} = "):
                        new_lines.append(f"{var_name} = {params}")
                        in_var = True
                    elif in_var and line.startswith('}'):
                        in_var = False
                    elif not in_var:
                        new_lines.append(line)
                
                # Réécrire le fichier
                with open(config_file, 'w') as f:
                    f.write('\n'.join(new_lines))
            else:
                # Ajouter les nouveaux paramètres à la fin du fichier
                with open(config_file, 'a') as f:
                    f.write(f"\n{var_name} = {params}\n")
            
            logger.info(f"Paramètres de la stratégie {strategy_name} sauvegardés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des paramètres de la stratégie: {e}")
            return False
    
    def optimize_strategy(self, strategy_name, symbol, timeframe, start_date=None, end_date=None,
                         optimization_method='grid', param_grid=None, n_iter=100, n_jobs=1):
        """
        Optimise les paramètres d'une stratégie
        
        Args:
            strategy_name (str): Nom de la stratégie
            symbol (str): Symbole de la paire
            timeframe (str): Timeframe
            start_date (datetime, optional): Date de début
            end_date (datetime, optional): Date de fin
            optimization_method (str, optional): Méthode d'optimisation ('grid' ou 'random')
            param_grid (dict, optional): Grille/distribution de paramètres à tester
            n_iter (int, optional): Nombre d'itérations pour la recherche aléatoire
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            pd.DataFrame: Résultats de l'optimisation
        """
        # Vérifier si la stratégie existe
        if strategy_name not in self.strategies:
            logger.error(f"Stratégie {strategy_name} inconnue")
            return None
            
        # Stratégie à optimiser
        strategy_class = self.strategies[strategy_name]
        
        # Dates par défaut
        if start_date is None:
            start_date = BACKTEST.get('start_date')
        if end_date is None:
            end_date = BACKTEST.get('end_date')
            
        # Paramètres par défaut de la stratégie
        default_params = self.load_strategy_params(strategy_name)
        
        # Grille/distribution de paramètres par défaut
        if param_grid is None:
            param_grid = self.get_default_param_grid(strategy_name, default_params)
            
        logger.info(f"Optimisation de la stratégie {strategy_name} pour {symbol} sur {timeframe}")
        logger.info(f"Période: {start_date} - {end_date}")
        
        # Initialiser l'optimiseur
        optimizer = StrategyOptimizer(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=CAPITAL.get('initial_capital', 10000),
            commission_rate=BACKTEST.get('commission_rate', 0.001),
            slippage=BACKTEST.get('slippage', 0.0005)
        )
        
        # Optimiser les paramètres
        if optimization_method == 'grid':
            results = optimizer.grid_search(param_grid, n_jobs=n_jobs)
        elif optimization_method == 'random':
            results = optimizer.random_search(param_grid, n_iter=n_iter, n_jobs=n_jobs)
        else:
            logger.error(f"Méthode d'optimisation {optimization_method} inconnue")
            return None
            
        # Sauvegarder les résultats
        if results is not None:
            result_file = self.results_dir / f"optimization_{strategy_name}_{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results.to_csv(result_file)
            logger.info(f"Résultats sauvegardés dans {result_file}")
            
        return results
    
    def update_strategy_params(self, strategy_name, best_params):
        """
        Met à jour les paramètres de la stratégie avec les meilleurs paramètres trouvés
        
        Args:
            strategy_name (str): Nom de la stratégie
            best_params (dict): Meilleurs paramètres
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        # Charger les paramètres actuels
        current_params = self.load_strategy_params(strategy_name)
        
        # Mettre à jour les paramètres
        for key, value in best_params.items():
            # Ne mettre à jour que les paramètres qui existaient déjà
            if key in current_params:
                current_params[key] = value
            
        # Sauvegarder les nouveaux paramètres
        return self.save_strategy_params(strategy_name, current_params)
    
    def get_default_param_grid(self, strategy_name, default_params):
        """
        Génère une grille de paramètres par défaut pour l'optimisation
        
        Args:
            strategy_name (str): Nom de la stratégie
            default_params (dict): Paramètres par défaut
            
        Returns:
            dict: Grille de paramètres
        """
        param_grid = {}
        
        if strategy_name == 'trend':
            param_grid = {
                'fast_ema': [5, 8, 10, 12, 15],
                'medium_ema': [15, 20, 25, 30, 35],
                'slow_ema': [40, 50, 60, 70, 80],
                'adx_threshold': [15, 20, 25, 30, 35],
                'buy_threshold': [0.3, 0.5, 0.7, 0.9],
                'sell_threshold': [-0.9, -0.7, -0.5, -0.3]
            }
        elif strategy_name == 'mean_reversion':
            param_grid = {
                'rsi_period': [7, 10, 14, 21],
                'rsi_overbought': [65, 70, 75, 80],
                'rsi_oversold': [20, 25, 30, 35],
                'bb_period': [15, 20, 25, 30],
                'bb_std': [1.5, 2.0, 2.5, 3.0],
                'buy_threshold': [0.3, 0.5, 0.7, 0.9],
                'sell_threshold': [-0.9, -0.7, -0.5, -0.3]
            }
        elif strategy_name == 'combined':
            param_grid = {
                'final_buy_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
                'final_sell_threshold': [-0.7, -0.6, -0.5, -0.4, -0.3],
                'adjust_weights_by_volatility': [True, False],
                'use_volume_filter': [True, False],
                'volume_threshold': [1.2, 1.5, 2.0, 2.5]
            }
        
        # Ne conserver que les paramètres qui existent dans les paramètres par défaut
        return {k: v for k, v in param_grid.items() if k in default_params}
    
    def auto_optimize_and_update(self, strategy_name, symbol, timeframe, lookback_days=180, 
                                optimization_method='grid', param_grid=None, n_iter=100, n_jobs=1):
        """
        Optimise automatiquement une stratégie et met à jour les paramètres
        
        Args:
            strategy_name (str): Nom de la stratégie
            symbol (str): Symbole de la paire
            timeframe (str): Timeframe
            lookback_days (int, optional): Nombre de jours de données à utiliser
            optimization_method (str, optional): Méthode d'optimisation ('grid' ou 'random')
            param_grid (dict, optional): Grille/distribution de paramètres à tester
            n_iter (int, optional): Nombre d'itérations pour la recherche aléatoire
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            dict: Résultats de l'optimisation
        """
        # Dates pour l'optimisation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Optimiser la stratégie
        results = self.optimize_strategy(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            optimization_method=optimization_method,
            param_grid=param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs
        )
        
        if results is None or len(results) == 0:
            logger.error("Aucun résultat d'optimisation, impossible de mettre à jour les paramètres")
            return None
            
        # Meilleurs paramètres
        best_params = {}
        for col in results.columns:
            if col not in ['profit_loss', 'profit_loss_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']:
                best_params[col] = results.iloc[0][col]
                
        # Mettre à jour les paramètres
        success = self.update_strategy_params(strategy_name, best_params)
        
        if success:
            logger.info(f"Paramètres de la stratégie {strategy_name} mis à jour avec succès")
            
            # Envoyer une notification
            subject = f"Mise à jour des paramètres de la stratégie {strategy_name}"
            message = f"Les paramètres de la stratégie {strategy_name} ont été mis à jour avec succès.\n\n"
            message += f"Meilleurs paramètres:\n"
            for key, value in best_params.items():
                message += f"- {key}: {value}\n"
                
            message += f"\nPerformance attendue:\n"
            message += f"- Rendement total: {results.iloc[0]['profit_loss_pct']:.2f}%\n"
            message += f"- Ratio de Sharpe: {results.iloc[0]['sharpe_ratio']:.2f}\n"
            message += f"- Drawdown maximum: {results.iloc[0]['max_drawdown']:.2f}%\n"
            message += f"- Taux de réussite: {results.iloc[0]['win_rate']:.2f}%\n"
            message += f"- Nombre de trades: {results.iloc[0]['total_trades']}\n"
            
            self.notifier.send_notification(subject, message)
            
        return {
            'success': success,
            'best_params': best_params,
            'performance': {
                'profit_loss_pct': results.iloc[0]['profit_loss_pct'],
                'sharpe_ratio': results.iloc[0]['sharpe_ratio'],
                'max_drawdown': results.iloc[0]['max_drawdown'],
                'win_rate': results.iloc[0]['win_rate'],
                'total_trades': results.iloc[0]['total_trades']
            }
        }
    
    def auto_optimize_all_strategies(self, symbol, timeframe, lookback_days=180, optimization_method='grid', n_jobs=1):
        """
        Optimise automatiquement toutes les stratégies disponibles
        
        Args:
            symbol (str): Symbole de la paire
            timeframe (str): Timeframe
            lookback_days (int, optional): Nombre de jours de données à utiliser
            optimization_method (str, optional): Méthode d'optimisation ('grid' ou 'random')
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            dict: Résultats de l'optimisation pour chaque stratégie
        """
        results = {}
        
        for strategy_name in self.strategies:
            logger.info(f"Optimisation de la stratégie {strategy_name}")
            
            result = self.auto_optimize_and_update(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=lookback_days,
                optimization_method=optimization_method,
                n_jobs=n_jobs
            )
            
            results[strategy_name] = result
            
        return results
    
    def schedule_optimization(self, strategy_name, symbol, timeframe, lookback_days=180, 
                            optimization_method='grid', frequency='weekly', day_of_week=6, hour=0):
        """
        Planifie une optimisation périodique
        
        Args:
            strategy_name (str): Nom de la stratégie
            symbol (str): Symbole de la paire
            timeframe (str): Timeframe
            lookback_days (int, optional): Nombre de jours de données à utiliser
            optimization_method (str, optional): Méthode d'optimisation ('grid' ou 'random')
            frequency (str, optional): Fréquence d'optimisation ('daily', 'weekly', 'monthly')
            day_of_week (int, optional): Jour de la semaine (0-6, 0=lundi)
            hour (int, optional): Heure du jour (0-23)
            
        Returns:
            bool: True si la planification a réussi, False sinon
        """
        # Cette méthode est un peu trompeuse car elle ne fait pas réellement de planification
        # Elle génère plutôt une commande crontab que l'utilisateur devra installer manuellement
        
        # Générer la commande crontab
        if frequency == 'daily':
            cron_expr = f"{hour} * * * *"
        elif frequency == 'weekly':
            cron_expr = f"{hour} * * {day_of_week} *"
        elif frequency == 'monthly':
            cron_expr = f"{hour} 1 * * *"
        else:
            logger.error(f"Fréquence {frequency} inconnue")
            return False
            
        # Chemin vers le script
        script_path = os.path.abspath(__file__)
        
        # Commande à exécuter
        command = f"python {script_path} --strategy {strategy_name} --symbol {symbol} --timeframe {timeframe} "
        command += f"--lookback {lookback_days} --method {optimization_method}"
        
        # Ligne crontab
        crontab_line = f"{cron_expr} {command}"
        
        logger.info(f"Pour planifier l'optimisation, ajoutez cette ligne à votre crontab:")
        logger.info(crontab_line)
        
        return True


# Fonction principale
def main():
    """
    Fonction principale
    """
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Auto-optimisation du bot de trading")
    
    parser.add_argument('--strategy', type=str, default='combined',
                       choices=['trend', 'mean_reversion', 'combined', 'all'],
                       help='Stratégie à optimiser')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Symbole de la paire')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe')
    
    parser.add_argument('--lookback', type=int, default=180,
                       help='Nombre de jours de données à utiliser')
    
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Méthode d\'optimisation')
    
    parser.add_argument('--iterations', type=int, default=100,
                       help='Nombre d\'itérations pour la recherche aléatoire')
    
    parser.add_argument('--jobs', type=int, default=1,
                       help='Nombre de processus parallèles')
    
    parser.add_argument('--schedule', action='store_true',
                       help='Planifier une optimisation périodique')
    
    parser.add_argument('--frequency', type=str, default='weekly',
                       choices=['daily', 'weekly', 'monthly'],
                       help='Fréquence d\'optimisation')
    
    args = parser.parse_args()
    
    # Initialiser l'auto-optimiseur
    optimizer = AutoOptimizer()
    
    # Planifier une optimisation périodique
    if args.schedule:
        if args.strategy == 'all':
            for strategy in ['trend', 'mean_reversion', 'combined']:
                optimizer.schedule_optimization(
                    strategy_name=strategy,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    lookback_days=args.lookback,
                    optimization_method=args.method,
                    frequency=args.frequency
                )
        else:
            optimizer.schedule_optimization(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                lookback_days=args.lookback,
                optimization_method=args.method,
                frequency=args.frequency
            )
            
        return
    
    # Optimiser une ou toutes les stratégies
    if args.strategy == 'all':
        results = optimizer.auto_optimize_all_strategies(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback,
            optimization_method=args.method,
            n_jobs=args.jobs
        )
        
        # Afficher les résultats
        for strategy_name, result in results.items():
            if result and result['success']:
                print(f"\nStratégie {strategy_name}:")
                print(f"Meilleurs paramètres: {result['best_params']}")
                print(f"Performance attendue:")
                print(f"- Rendement total: {result['performance']['profit_loss_pct']:.2f}%")
                print(f"- Ratio de Sharpe: {result['performance']['sharpe_ratio']:.2f}")
                print(f"- Drawdown maximum: {result['performance']['max_drawdown']:.2f}%")
                print(f"- Taux de réussite: {result['performance']['win_rate']:.2f}%")
                print(f"- Nombre de trades: {result['performance']['total_trades']}")
    else:
        result = optimizer.auto_optimize_and_update(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback,
            optimization_method=args.method,
            n_iter=args.iterations,
            n_jobs=args.jobs
        )
        
        # Afficher les résultats
        if result and result['success']:
            print(f"\nStratégie {args.strategy}:")
            print(f"Meilleurs paramètres: {result['best_params']}")
            print(f"Performance attendue:")
            print(f"- Rendement total: {result['performance']['profit_loss_pct']:.2f}%")
            print(f"- Ratio de Sharpe: {result['performance']['sharpe_ratio']:.2f}")
            print(f"- Drawdown maximum: {result['performance']['max_drawdown']:.2f}%")
            print(f"- Taux de réussite: {result['performance']['win_rate']:.2f}%")
            print(f"- Nombre de trades: {result['performance']['total_trades']}")


# Point d'entrée
if __name__ == "__main__":
    main()