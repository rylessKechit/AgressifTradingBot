"""
Script de backtesting
Permet de tester les stratégies de trading sur des données historiques
"""
import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules nécessaires
from config.settings import BACKTEST, TRADING, CAPITAL
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.combined_strategy import CombinedStrategy
from execution.trader import Trader
from risk.position_sizing import PositionSizer
from utils.logger import setup_logger
from utils.visualizer import plot_backtest_results

# Configuration du logging
logger = setup_logger(level=logging.INFO)

class Backtester:
    """
    Classe pour effectuer le backtesting des stratégies de trading
    """
    
    def __init__(self, strategy_name="combined", start_date=None, end_date=None, 
                initial_capital=None, commission_rate=None, slippage=None):
        """
        Initialise le backtester
        
        Args:
            strategy_name (str, optional): Nom de la stratégie ("trend", "mean_reversion", "combined")
            start_date (datetime, optional): Date de début du backtest
            end_date (datetime, optional): Date de fin du backtest
            initial_capital (float, optional): Capital initial
            commission_rate (float, optional): Taux de commission
            slippage (float, optional): Slippage
        """
        self.strategy_name = strategy_name
        self.start_date = start_date or BACKTEST.get('start_date')
        self.end_date = end_date or BACKTEST.get('end_date')
        self.initial_capital = initial_capital or CAPITAL.get('initial_capital', 10000)
        self.commission_rate = commission_rate or BACKTEST.get('commission_rate', 0.001)
        self.slippage = slippage or BACKTEST.get('slippage', 0.0005)
        
        # Initialiser les composants
        self.fetcher = DataFetcher()
        self.position_sizer = PositionSizer(
            initial_capital=self.initial_capital,
            max_risk_per_trade=CAPITAL.get('risk_per_trade', 0.01),
            max_position_size=CAPITAL.get('max_position_size', 0.2)
        )
        self.trader = Trader(position_sizer=self.position_sizer, mode="backtest")
        
        # Initialiser la stratégie
        self.strategy = self._initialize_strategy()
        
        logger.info(f"Backtester initialisé avec la stratégie {self.strategy_name}")
        logger.info(f"Période: {self.start_date} - {self.end_date}")
        logger.info(f"Capital initial: {self.initial_capital}")
        logger.info(f"Commission: {self.commission_rate * 100}%, Slippage: {self.slippage * 100}%")
    
    def _initialize_strategy(self):
        """
        Initialise la stratégie de trading
        
        Returns:
            BaseStrategy: Stratégie de trading
        """
        if self.strategy_name == "trend":
            return TrendFollowingStrategy()
        elif self.strategy_name == "mean_reversion":
            return MeanReversionStrategy()
        else:  # combined
            return CombinedStrategy()
    
    def run(self, symbol=None, timeframe=None):
        """
        Exécute le backtest
        """
        # Utiliser les valeurs par défaut si non spécifiées
        symbol = symbol or TRADING.get('trading_pairs', ['BTC/USDT'])[0]
        timeframe = timeframe or TRADING.get('timeframe', '15m')  # Changer à 15m par défaut
        
        logger.info(f"Démarrage du backtest pour {symbol} sur {timeframe}")
        
        try:
            # Récupérer les données historiques
            data = self._fetch_data(symbol, timeframe)
            
            if data is None or data.empty:
                logger.error(f"Pas de données pour {symbol} sur {timeframe}")
                return None
            
            # Afficher des informations sur les données
            logger.info(f"Données récupérées: {len(data)} lignes de {data.index.min()} à {data.index.max()}")
            logger.info(f"Plage de prix: {data['low'].min():.2f} - {data['high'].max():.2f}")
            
            # Ajouter les indicateurs techniques
            data = TechnicalIndicators.add_all_indicators(data)
            
            # Vérifier que les indicateurs importants sont présents
            required_indicators = [
                'ema_8', 'ema_21', 'macd', 'macd_signal', 'rsi_14', 
                'bb_upper', 'bb_middle', 'bb_lower'
            ]
            
            missing = [ind for ind in required_indicators if ind not in data.columns]
            if missing:
                logger.error(f"Indicateurs manquants: {missing}")
                return None
                
            # Générer les signaux
            signals = self.strategy.run(data)
            
            # Vérifier si des signaux ont été générés
            if signals is None or signals.empty:
                logger.error(f"Pas de signaux générés pour {symbol} sur {timeframe}")
                return None
                
            num_signals = (signals != 0).sum()
            if num_signals == 0:
                logger.error(f"Aucun signal non-nul généré pour {symbol} sur {timeframe}")
                
                # Debug: vérifier les croisements possibles
                cross_up = ((data['ema_8'] > data['ema_21']) & 
                        (data['ema_8'].shift(1) <= data['ema_21'].shift(1))).sum()
                cross_down = ((data['ema_8'] < data['ema_21']) & 
                            (data['ema_8'].shift(1) >= data['ema_21'].shift(1))).sum()
                            
                logger.info(f"Croisements EMA 8/21: {cross_up} haussiers, {cross_down} baissiers")
                
                # Vérifier les zones de surachat/survente RSI
                rsi_overbought = (data['rsi_14'] > 70).sum()
                rsi_oversold = (data['rsi_14'] < 30).sum()
                
                logger.info(f"RSI: {rsi_oversold} périodes de survente, {rsi_overbought} périodes de surachat")
                
                return None
                
            logger.info(f"{num_signals} signaux générés: {(signals == 1).sum()} achats, {(signals == -1).sum()} ventes")
            
            # Exécuter le backtest
            results = self._execute_backtest(symbol, data, signals)
            
            # Afficher les résultats
            if results:
                self._print_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _fetch_data(self, symbol, timeframe):
        """
        Récupère les données historiques
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            timeframe (str): Timeframe (ex: 15m, 1h, 4h)
            
        Returns:
            pd.DataFrame: Données historiques
        """
        try:
            # Récupérer les données
            data = self.fetcher.fetch_historical_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Vérifier si les données sont vides
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                logger.error(f"Aucune donnée récupérée pour {symbol} sur {timeframe}")
                return pd.DataFrame()  # Retourner un DataFrame vide mais initialisé
            
            # Si les données sont retournées sous forme de liste
            if isinstance(data, list):
                if len(data) == 0:
                    logger.error(f"Liste de données vide pour {symbol} sur {timeframe}")
                    return pd.DataFrame()
                    
                # Convertir en DataFrame
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                df = data
                    
            logger.info(f"Données récupérées: {len(df)} bougies de {df.index.min() if not df.empty else 'N/A'} à {df.index.max() if not df.empty else 'N/A'}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()  # Retourner un DataFrame vide mais initialisé

    def _execute_backtest(self, symbol, data, signals):
        """
        Exécute le backtest
        
        Args:
            symbol (str): Symbole de la paire
            data (pd.DataFrame): Données historiques
            signals (pd.Series): Signaux de trading
            
        Returns:
            dict: Résultats du backtest
        """
        # Vérifier si les données sont vides
        if data.empty or len(data) == 0:
            logger.error(f"Données vides pour {symbol}, impossible d'exécuter le backtest")
            return None
            
        # Réinitialiser le trader
        self.trader = Trader(position_sizer=self.position_sizer, mode="backtest")
        
        # Vérifier s'il y a des données pour l'historique d'équité
        if len(data.index) == 0:
            logger.error("Aucune donnée disponible pour le backtest")
            return None
        
        # Historique de l'équité
        equity_history = [self.initial_capital]
        dates = [data.index[0]]  # Assurez-vous que data a au moins une ligne
        
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
                self.trader.execute_signal(symbol, current_signal, current_data)
            
            # Mettre à jour les trailing stops
            for sym in list(self.trader.active_positions.keys()):
                self.trader.update_trailing_stops(sym, current_price)
            
            # Mettre à jour l'équité
            equity_history.append(self.trader.current_capital)
            dates.append(current_date)
        
        # Calculer les résultats
        stats = self.trader.get_stats()
        trades = self.trader.get_trades_history()
        
        # Créer la courbe d'équité
        equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity_history
        }).set_index('date')
        
        # Ajouter des statistiques supplémentaires
        stats['equity_curve'] = equity_curve
        stats['trades'] = trades
        stats['symbol'] = symbol
        stats['timeframe'] = data.index[1] - data.index[0]  # Approximation du timeframe
        stats['start_date'] = data.index[0]
        stats['end_date'] = data.index[-1]
        stats['duration'] = data.index[-1] - data.index[0]
        stats['signals'] = signals
        
        # Calculer des métriques de performance
        stats.update(self._calculate_performance_metrics(equity_curve, trades))
        
        return stats
    
    def _calculate_performance_metrics(self, equity_curve, trades):
        """
        Calcule des métriques de performance supplémentaires
        
        Args:
            equity_curve (pd.DataFrame): Courbe d'équité
            trades (list): Historique des trades
            
        Returns:
            dict: Métriques de performance
        """
        metrics = {}
        
        # Calculer les rendements quotidiens
        if not equity_curve.empty:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()
            
            # Rendement annualisé
            total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            total_years = total_days / 365.25
            
            if total_years > 0:
                total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1
                metrics['annualized_return'] = (1 + total_return) ** (1 / total_years) - 1
            else:
                metrics['annualized_return'] = 0
            
            # Volatilité
            metrics['volatility'] = equity_curve['daily_return'].std() * np.sqrt(252)
            
            # Ratio de Sharpe (0% comme taux sans risque)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
                
            # Maximum Drawdown
            equity_curve['cum_max'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['cum_max'] - equity_curve['equity']) / equity_curve['cum_max']
            metrics['max_drawdown'] = equity_curve['drawdown'].max()
            
            # Calmar Ratio
            if metrics['max_drawdown'] > 0:
                metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
            else:
                metrics['calmar_ratio'] = 0
        
        # Statistiques des trades
        if trades:
            # Gains/pertes moyens
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            metrics['avg_profit'] = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Profit Factor
            total_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            total_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
            
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Durée moyenne des trades
            durations = [(t['exit_time'] - t['entry_time']).total_seconds() / (60 * 60) for t in trades]  # en heures
            metrics['avg_trade_duration'] = np.mean(durations) if durations else 0
            
            # Gains consécutifs/pertes consécutives
            metrics['max_consecutive_wins'] = self._max_consecutive(trades, True)
            metrics['max_consecutive_losses'] = self._max_consecutive(trades, False)
        
        return metrics
    
    def _max_consecutive(self, trades, winning=True):
        """
        Calcule le nombre maximum de trades consécutifs gagnants ou perdants
        
        Args:
            trades (list): Historique des trades
            winning (bool, optional): Calculer les trades gagnants (True) ou perdants (False)
            
        Returns:
            int: Nombre maximum de trades consécutifs
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            is_winning = trade['pnl'] > 0
            
            if is_winning == winning:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def _print_results(self, results):
        """
        Affiche les résultats du backtest
        
        Args:
            results (dict): Résultats du backtest
        """
        if not results:
            return
        
        print("\n==== RÉSULTATS DU BACKTEST ====")
        print(f"Symbole: {results['symbol']}")
        print(f"Période: {results['start_date']} - {results['end_date']} ({results['duration']})")
        print(f"Stratégie: {self.strategy_name}")
        print("\n--- PERFORMANCE ---")
        print(f"Capital initial: {self.initial_capital:.2f}")
        print(f"Capital final: {results['current_capital']:.2f}")
        print(f"Profit/Perte: {results['profit_loss']:.2f} ({results['profit_loss_pct']:.2f}%)")
        print(f"Rendement annualisé: {results.get('annualized_return', 0) * 100:.2f}%")
        print(f"Ratio de Sharpe: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Ratio de Calmar: {results.get('calmar_ratio', 0):.2f}")
        print(f"Drawdown maximum: {results.get('max_drawdown', 0) * 100:.2f}%")
        print("\n--- TRADES ---")
        print(f"Nombre de trades: {results['total_trades']}")
        print(f"Trades gagnants: {results['winning_trades']} ({results['win_rate']:.2f}%)")
        print(f"Trades perdants: {results['losing_trades']}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Gain moyen: {results.get('avg_profit', 0):.2f}")
        print(f"Perte moyenne: {results.get('avg_loss', 0):.2f}")
        print(f"Gains consécutifs max: {results.get('max_consecutive_wins', 0)}")
        print(f"Pertes consécutives max: {results.get('max_consecutive_losses', 0)}")
        print(f"Durée moyenne des trades: {results.get('avg_trade_duration', 0):.2f} heures")
    
    def optimize_parameters(self, symbol, timeframe, param_grid):
        """
        Optimise les paramètres de la stratégie
        
        Args:
            symbol (str): Symbole de la paire
            timeframe (str): Timeframe
            param_grid (dict): Grille de paramètres à tester
            
        Returns:
            pd.DataFrame: Résultats de l'optimisation
        """
        # Récupérer les données
        data = self._fetch_data(symbol, timeframe)
        
        if data is None or data.empty:
            logger.error(f"Pas de données pour {symbol} sur {timeframe}")
            return None
            
        # Ajouter les indicateurs techniques
        data = TechnicalIndicators.add_all_indicators(data)
        
        # Générer toutes les combinaisons de paramètres
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        # Tester chaque combinaison
        for combo in combinations:
            # Créer le dictionnaire de paramètres
            params = dict(zip(param_keys, combo))
            
            # Mettre à jour les paramètres de la stratégie
            self.strategy.set_parameters(params)
            
            # Générer les signaux
            signals = self.strategy.run(data)
            
            # Exécuter le backtest
            backtest_results = self._execute_backtest(symbol, data, signals)
            
            # Ajouter les résultats
            result = {
                'params': params,
                'profit_loss': backtest_results['profit_loss'],
                'profit_loss_pct': backtest_results['profit_loss_pct'],
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'win_rate': backtest_results['win_rate'],
                'total_trades': backtest_results['total_trades'],
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'profit_factor': backtest_results.get('profit_factor', 0)
            }
            
            # Ajouter les paramètres au résultat
            for key, value in params.items():
                result[key] = value
                
            results.append(result)
            
            logger.info(f"Testé: {params} - Profit: {result['profit_loss_pct']:.2f}%, Trades: {result['total_trades']}")
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        # Trier par profit
        results_df = results_df.sort_values('profit_loss_pct', ascending=False)
        
        return results_df
    
    def plot_results(self, results, show_trades=True, show_drawdown=True):
        """
        Trace les résultats du backtest
        
        Args:
            results (dict): Résultats du backtest
            show_trades (bool, optional): Afficher les trades
            show_drawdown (bool, optional): Afficher le drawdown
        """
        # Tracer avec la fonction du module utils.visualizer
        plot_backtest_results(results, show_trades, show_drawdown)


# Fonction principale
def main():
    """
    Fonction principale
    """
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description='Backtesting de stratégies de trading')
    
    parser.add_argument('--strategy', type=str, default='combined',
                       choices=['trend', 'mean_reversion', 'combined'],
                       help='Stratégie de trading à tester')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Symbole de la paire')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe')
    
    parser.add_argument('--start', type=str, default=None,
                       help='Date de début (format: YYYY-MM-DD)')
    
    parser.add_argument('--end', type=str, default=None,
                       help='Date de fin (format: YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=10000,
                       help='Capital initial')
    
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Taux de commission')
    
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='Slippage')
    
    parser.add_argument('--optimize', action='store_true',
                       help='Activer l\'optimisation des paramètres')
    
    args = parser.parse_args()
    
    # Convertir les dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
    
    # Initialiser le backtester
    backtester = Backtester(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage=args.slippage
    )
    
    # Optimisation des paramètres
    if args.optimize:
        # Exemple de grille de paramètres à optimiser
        param_grid = {}
        
        if args.strategy == 'trend':
            param_grid = {
                'fast_ema': [8, 10, 12],
                'slow_ema': [20, 30, 40],
                'very_slow_ema': [150, 200, 250],
                'adx_threshold': [20, 25, 30],
                'buy_threshold': [0.5, 0.7, 0.9],
                'sell_threshold': [-0.5, -0.7, -0.9]
            }
        elif args.strategy == 'mean_reversion':
            param_grid = {
                'rsi_period': [10, 14, 18],
                'rsi_overbought': [65, 70, 75],
                'rsi_oversold': [25, 30, 35],
                'bb_period': [15, 20, 25],
                'buy_threshold': [0.5, 0.7, 0.9],
                'sell_threshold': [-0.5, -0.7, -0.9]
            }
        else:  # combined
            param_grid = {
                'final_buy_threshold': [0.3, 0.5, 0.7],
                'final_sell_threshold': [-0.3, -0.5, -0.7]
            }
        
        # Exécuter l'optimisation
        results = backtester.optimize_parameters(args.symbol, args.timeframe, param_grid)
        
        # Afficher les meilleurs résultats
        print("\n==== RÉSULTATS DE L'OPTIMISATION ====")
        print(results.head(10))
        
        # Enregistrer les résultats
        results.to_csv(f"optimization_results_{args.strategy}_{args.symbol.replace('/', '_')}_{args.timeframe}.csv")
        
    else:
        # Exécuter le backtest
        results = backtester.run(args.symbol, args.timeframe)
        
        # Tracer les résultats
        if results:
            backtester.plot_results(results)


# Point d'entrée
if __name__ == "__main__":
    main()