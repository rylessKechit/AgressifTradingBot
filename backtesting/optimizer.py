"""
Module d'optimisation de stratégies
Fournit des méthodes pour optimiser les paramètres des stratégies de trading
"""
import os
import sys
import time
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BACKTEST, CAPITAL
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from execution.trader import Trader
from risk.position_sizing import PositionSizer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Classe pour optimiser les paramètres d'une stratégie de trading
    """
    
    def __init__(self, strategy_class, symbol, timeframe, start_date=None, end_date=None,
                initial_capital=None, commission_rate=None, slippage=None):
        """
        Initialise l'optimiseur de stratégie
        
        Args:
            strategy_class (class): Classe de la stratégie à optimiser
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            timeframe (str): Timeframe (ex: 15m, 1h, 4h)
            start_date (datetime, optional): Date de début
            end_date (datetime, optional): Date de fin
            initial_capital (float, optional): Capital initial
            commission_rate (float, optional): Taux de commission
            slippage (float, optional): Slippage
        """
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date or BACKTEST.get('start_date')
        self.end_date = end_date or BACKTEST.get('end_date')
        self.initial_capital = initial_capital or CAPITAL.get('initial_capital', 10000)
        self.commission_rate = commission_rate or BACKTEST.get('commission_rate', 0.001)
        self.slippage = slippage or BACKTEST.get('slippage', 0.0005)
        
        # Initialiser les composants nécessaires
        self.fetcher = DataFetcher()
        self.position_sizer = PositionSizer(
            initial_capital=self.initial_capital,
            max_risk_per_trade=CAPITAL.get('risk_per_trade', 0.01),
            max_position_size=CAPITAL.get('max_position_size', 0.2)
        )
        
        # Charger les données historiques
        self.data = self._load_data()
        
        # Préparer le trader pour le backtesting
        self.trader = Trader(position_sizer=self.position_sizer, mode="backtest")
        
        logger.info(f"Optimiseur initialisé pour {self.symbol} sur {self.timeframe} de {self.start_date} à {self.end_date}")
    
    def _load_data(self):
        """
        Charge les données historiques et ajoute les indicateurs
        
        Returns:
            pd.DataFrame: Données historiques avec indicateurs
        """
        try:
            # Récupérer les données
            data = self.fetcher.fetch_historical_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Convertir en DataFrame si nécessaire
            if isinstance(data, list):
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                df = data
            
            # Ajouter les indicateurs techniques
            df = TechnicalIndicators.add_all_indicators(df)
            
            logger.info(f"Données chargées et indicateurs calculés: {len(df)} bougies")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return None
    
    def _evaluate_params(self, params):
        """
        Évalue une combinaison de paramètres
        
        Args:
            params (dict): Paramètres à évaluer
            
        Returns:
            dict: Résultats de l'évaluation
        """
        try:
            # Initialiser la stratégie avec les paramètres
            strategy = self.strategy_class(params)
            
            # Générer les signaux
            signals = strategy.run(self.data)
            
            # Réinitialiser le trader
            self.trader = Trader(position_sizer=self.position_sizer, mode="backtest")
            
            # Exécuter le backtest
            equity_history = [self.initial_capital]
            dates = [self.data.index[0]]
            
            # Exécuter les trades
            for i in range(1, len(self.data)):
                # Date actuelle
                current_date = self.data.index[i]
                
                # Données jusqu'à l'index actuel
                current_data = self.data.iloc[:i+1]
                
                # Signal actuel
                current_signal = signals.iloc[i]
                
                # Prix actuel
                current_price = self.data['close'].iloc[i]
                
                # Exécuter le signal
                if current_signal != 0:
                    self.trader.execute_signal(self.symbol, current_signal, current_data)
                
                # Mettre à jour les trailing stops
                for sym in list(self.trader.active_positions.keys()):
                    self.trader.update_trailing_stops(sym, current_price)
                
                # Mettre à jour l'équité
                equity_history.append(self.trader.current_capital)
                dates.append(current_date)
            
            # Récupérer les statistiques
            stats = self.trader.get_stats()
            
            # Calculer des métriques supplémentaires
            equity_curve = pd.DataFrame({
                'date': dates,
                'equity': equity_history
            }).set_index('date')
            
            # Calcul du ratio de Sharpe
            if len(equity_curve) > 1:
                equity_curve['returns'] = equity_curve['equity'].pct_change()
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
            
            # Résultat
            result = {
                'params': params,
                'profit_loss': stats.get('profit_loss', 0),
                'profit_loss_pct': stats.get('profit_loss_pct', 0),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats.get('total_trades', 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des paramètres {params}: {e}")
            return {
                'params': params,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 1,
                'win_rate': 0,
                'total_trades': 0
            }
    
    def grid_search(self, param_grid, n_jobs=1):
        """
        Effectue une recherche en grille pour trouver les meilleurs paramètres
        
        Args:
            param_grid (dict): Grille de paramètres à tester
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            pd.DataFrame: Résultats de la recherche en grille
        """
        if self.data is None or self.data.empty:
            logger.error("Pas de données disponibles pour l'optimisation")
            return None
        
        # Générer toutes les combinaisons de paramètres
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Début de la recherche en grille avec {len(combinations)} combinaisons")
        start_time = time.time()
        
        results = []
        
        # Exécuter en parallèle si demandé
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                
                for combo in combinations:
                    params = dict(zip(param_keys, combo))
                    futures.append(executor.submit(self._evaluate_params, params))
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Optimisation"):
                    result = future.result()
                    results.append(result)
        else:
            # Exécution séquentielle
            for combo in tqdm(combinations, desc="Optimisation"):
                params = dict(zip(param_keys, combo))
                result = self._evaluate_params(params)
                results.append(result)
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        # Ajouter les colonnes de paramètres
        for i, param_name in enumerate(param_keys):
            results_df[param_name] = results_df['params'].apply(lambda x: x.get(param_name))
        
        # Supprimer la colonne params
        results_df = results_df.drop('params', axis=1)
        
        # Trier par profit décroissant
        results_df = results_df.sort_values('profit_loss_pct', ascending=False)
        
        # Afficher le temps d'exécution
        elapsed_time = time.time() - start_time
        logger.info(f"Recherche en grille terminée en {elapsed_time:.2f} secondes")
        
        return results_df
    
    def random_search(self, param_distributions, n_iter=100, n_jobs=1):
        """
        Effectue une recherche aléatoire pour trouver les meilleurs paramètres
        
        Args:
            param_distributions (dict): Distributions des paramètres à tester
            n_iter (int, optional): Nombre d'itérations
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            pd.DataFrame: Résultats de la recherche aléatoire
        """
        if self.data is None or self.data.empty:
            logger.error("Pas de données disponibles pour l'optimisation")
            return None
        
        logger.info(f"Début de la recherche aléatoire avec {n_iter} itérations")
        start_time = time.time()
        
        # Générer n_iter combinaisons aléatoires de paramètres
        param_combinations = []
        for _ in range(n_iter):
            params = {}
            for param_name, param_dist in param_distributions.items():
                if isinstance(param_dist, list):
                    # Choix aléatoire dans une liste
                    params[param_name] = np.random.choice(param_dist)
                elif isinstance(param_dist, tuple) and len(param_dist) == 2:
                    # Distribution uniforme entre deux valeurs
                    low, high = param_dist
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = np.random.randint(low, high + 1)
                    else:
                        params[param_name] = np.random.uniform(low, high)
                else:
                    # Valeur fixe
                    params[param_name] = param_dist
            
            param_combinations.append(params)
        
        results = []
        
        # Exécuter en parallèle si demandé
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                
                for params in param_combinations:
                    futures.append(executor.submit(self._evaluate_params, params))
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Optimisation"):
                    result = future.result()
                    results.append(result)
        else:
            # Exécution séquentielle
            for params in tqdm(param_combinations, desc="Optimisation"):
                result = self._evaluate_params(params)
                results.append(result)
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        # Ajouter les colonnes de paramètres
        for param_name in param_distributions.keys():
            results_df[param_name] = results_df['params'].apply(lambda x: x.get(param_name))
        
        # Supprimer la colonne params
        results_df = results_df.drop('params', axis=1)
        
        # Trier par profit décroissant
        results_df = results_df.sort_values('profit_loss_pct', ascending=False)
        
        # Afficher le temps d'exécution
        elapsed_time = time.time() - start_time
        logger.info(f"Recherche aléatoire terminée en {elapsed_time:.2f} secondes")
        
        return results_df
    
    def walk_forward_optimization(self, param_grid, window_size=90, step_size=30, n_jobs=1):
        """
        Effectue une optimisation walk-forward pour trouver les meilleurs paramètres
        
        Args:
            param_grid (dict): Grille de paramètres à tester
            window_size (int, optional): Taille de la fenêtre d'optimisation en jours
            step_size (int, optional): Taille du pas en jours
            n_jobs (int, optional): Nombre de processus parallèles
            
        Returns:
            tuple: (DataFrame des résultats, DataFrame des paramètres optimaux par période)
        """
        if self.data is None or self.data.empty:
            logger.error("Pas de données disponibles pour l'optimisation")
            return None, None
        
        logger.info(f"Début de l'optimisation walk-forward avec fenêtre={window_size}, pas={step_size}")
        
        # Calculer les périodes de temps
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        
        # Calculer les périodes d'optimisation et de test
        periods = []
        current_date = start_date
        
        while current_date < end_date:
            opt_start = current_date
            opt_end = opt_start + pd.Timedelta(days=window_size)
            test_start = opt_end
            test_end = test_start + pd.Timedelta(days=step_size)
            
            if test_end > end_date:
                test_end = end_date
                
            periods.append({
                'opt_start': opt_start,
                'opt_end': opt_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date = test_start
        
        results = []
        optimal_params = []
        
        # Pour chaque période
        for i, period in enumerate(periods):
            logger.info(f"Période {i+1}/{len(periods)}: Optimisation {period['opt_start']} - {period['opt_end']}, Test {period['test_start']} - {period['test_end']}")
            
            # Données d'optimisation
            opt_data = self.data[(self.data.index >= period['opt_start']) & (self.data.index < period['opt_end'])]
            
            if len(opt_data) < 30:  # Nombre minimum de bougies
                logger.warning(f"Pas assez de données pour la période d'optimisation {i+1}, ignorée")
                continue
                
            # Optimisation sur la période d'optimisation
            optimizer = StrategyOptimizer(
                strategy_class=self.strategy_class,
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=period['opt_start'],
                end_date=period['opt_end'],
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage=self.slippage
            )
            
            # Recherche en grille
            opt_results = optimizer.grid_search(param_grid, n_jobs=n_jobs)
            
            if opt_results is None or len(opt_results) == 0:
                logger.warning(f"Pas de résultats pour la période d'optimisation {i+1}, ignorée")
                continue
                
            # Meilleurs paramètres
            best_params = {col: opt_results.iloc[0][col] for col in param_grid.keys()}
            
            # Ajouter à la liste des paramètres optimaux
            optimal_params.append({
                'period': i + 1,
                'opt_start': period['opt_start'],
                'opt_end': period['opt_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                **best_params
            })
            
            # Test sur la période de test
            test_data = self.data[(self.data.index >= period['test_start']) & (self.data.index < period['test_end'])]
            
            if len(test_data) < 5:  # Nombre minimum de bougies
                logger.warning(f"Pas assez de données pour la période de test {i+1}, ignorée")
                continue
                
            # Initialiser la stratégie avec les meilleurs paramètres
            strategy = self.strategy_class(best_params)
            
            # Générer les signaux
            signals = strategy.run(test_data)
            
            # Réinitialiser le trader
            trader = Trader(position_sizer=self.position_sizer, mode="backtest")
            
            # Exécuter le backtest
            equity_history = [self.initial_capital]
            dates = [test_data.index[0]]
            
            # Exécuter les trades
            for j in range(1, len(test_data)):
                # Date actuelle
                current_date = test_data.index[j]
                
                # Données jusqu'à l'index actuel
                current_data = test_data.iloc[:j+1]
                
                # Signal actuel
                current_signal = signals.iloc[j] if j < len(signals) else 0
                
                # Prix actuel
                current_price = test_data['close'].iloc[j]
                
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
            
            # Résultat pour cette période
            result = {
                'period': i + 1,
                'opt_start': period['opt_start'],
                'opt_end': period['opt_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'profit_loss': stats.get('profit_loss', 0),
                'profit_loss_pct': stats.get('profit_loss_pct', 0),
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats.get('total_trades', 0),
                **best_params
            }
            
            results.append(result)
            
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        optimal_params_df = pd.DataFrame(optimal_params)
        
        return results_df, optimal_params_df
    
    def save_results(self, results, filename):
        """
        Sauvegarde les résultats dans un fichier CSV
        
        Args:
            results (pd.DataFrame): Résultats à sauvegarder
            filename (str): Nom du fichier
        """
        if results is None or len(results) == 0:
            logger.warning("Pas de résultats à sauvegarder")
            return
            
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Sauvegarder
        results.to_csv(filename)
        logger.info(f"Résultats sauvegardés dans {filename}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logger = setup_logger(level=logging.INFO)
    
    from strategies.trend_following import TrendFollowingStrategy
    
    # Initialiser l'optimiseur
    optimizer = StrategyOptimizer(
        strategy_class=TrendFollowingStrategy,
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1)
    )
    
    # Exemple de grille de paramètres
    param_grid = {
        'fast_ema': [8, 10, 12],
        'medium_ema': [20, 25, 30],
        'slow_ema': [40, 50, 60],
        'adx_threshold': [20, 25, 30],
        'buy_threshold': [0.5, 0.7],
        'sell_threshold': [-0.5, -0.7]
    }
    
    # Recherche en grille
    results = optimizer.grid_search(param_grid, n_jobs=4)
    
    # Afficher les meilleurs résultats
    if results is not None:
        print("\nMeilleurs paramètres:")
        print(results.head(5))
        
        # Sauvegarder les résultats
        optimizer.save_results(results, "results/grid_search_results.csv")
    
    # Exemple de recherche aléatoire
    param_distributions = {
        'fast_ema': (5, 15),
        'medium_ema': (15, 35),
        'slow_ema': (30, 70),
        'adx_threshold': (15, 35),
        'buy_threshold': (0.3, 0.9),
        'sell_threshold': (-0.9, -0.3)
    }
    
    # Recherche aléatoire
    random_results = optimizer.random_search(param_distributions, n_iter=50, n_jobs=4)
    
    # Afficher les meilleurs résultats
    if random_results is not None:
        print("\nMeilleurs paramètres (recherche aléatoire):")
        print(random_results.head(5))
        
        # Sauvegarder les résultats
        optimizer.save_results(random_results, "results/random_search_results.csv")
    
    # Exemple d'optimisation walk-forward
    wf_results, optimal_params = optimizer.walk_forward_optimization(param_grid, window_size=60, step_size=30, n_jobs=4)
    
    # Afficher les résultats
    if wf_results is not None:
        print("\nRésultats de l'optimisation walk-forward:")
        print(wf_results)
        
        print("\nParamètres optimaux par période:")
        print(optimal_params)
        
        # Sauvegarder les résultats
        optimizer.save_results(wf_results, "results/walk_forward_results.csv")
        optimizer.save_results(optimal_params, "results/walk_forward_params.csv")