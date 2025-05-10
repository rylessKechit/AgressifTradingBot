"""
Point d'entrée principal du bot de trading
Lance le bot en mode trading en direct
"""
import os
import sys
import time
import logging
import signal
from datetime import datetime, timedelta
import pandas as pd

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules nécessaires
from config.settings import TRADING, CAPITAL, LOG_LEVEL, LOG_FORMAT, LOG_FILE, EXECUTION_MODE
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.combined_strategy import CombinedStrategy
from execution.exchange import Exchange
from execution.trader import Trader
from risk.position_sizing import PositionSizer
from utils.logger import setup_logger
from utils.email_notifier import EmailNotifier

# Configuration du logging
logger = setup_logger(LOG_LEVEL, LOG_FORMAT, LOG_FILE)

class TradingBot:
    """
    Classe principale du bot de trading
    """
    
    def __init__(self, mode=EXECUTION_MODE):
        """
        Initialise le bot de trading
        
        Args:
            mode (str, optional): Mode d'exécution ("backtest", "paper", "live")
        """
        self.mode = mode
        logger.info(f"Initialisation du bot en mode {mode}")
        
        # Initialiser les composants
        self.exchange = Exchange()
        self.fetcher = DataFetcher()
        self.position_sizer = PositionSizer(
            initial_capital=CAPITAL.get('initial_capital', 10000),
            max_risk_per_trade=CAPITAL.get('risk_per_trade', 0.01),
            max_position_size=CAPITAL.get('max_position_size', 0.2)
        )
        self.trader = Trader(self.exchange, self.position_sizer, mode=mode)
        
        # Initialiser la stratégie
        self.strategy = self._initialize_strategy()
        
        # Initialiser les notifications
        self.notifier = EmailNotifier()
        
        # Variables de contrôle
        self.running = False
        self.last_check_time = {}
        
        # Configurer le gestionnaire de signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Bot de trading initialisé avec succès")
    
    def _initialize_strategy(self):
        """
        Initialise la stratégie de trading
        
        Returns:
            BaseStrategy: Stratégie de trading
        """
        # Utiliser la stratégie combinée par défaut
        strategy = CombinedStrategy()
        logger.info(f"Stratégie initialisée: {strategy.name}")
        return strategy
    
    def _signal_handler(self, signum, frame):
        """
        Gestionnaire de signal pour arrêter proprement le bot
        
        Args:
            signum: Numéro du signal
            frame: Frame courante
        """
        logger.info(f"Signal reçu: {signum}, arrêt du bot")
        self.stop()
    
    def start(self):
        """
        Démarre le bot de trading
        """
        if self.running:
            logger.warning("Le bot est déjà en cours d'exécution")
            return
            
        self.running = True
        logger.info("Démarrage du bot de trading")
        
        try:
            # Vérifier le solde du compte
            balance = self.exchange.get_balance()
            if balance:
                logger.info(f"Solde du compte: {balance['total']}")
            
            # Boucle principale
            while self.running:
                # Vérifier et exécuter les trades pour chaque symbole
                for symbol in TRADING.get('trading_pairs', []):
                    self._check_and_execute(symbol)
                
                # Mettre à jour les trailing stops
                self._update_trailing_stops()
                
                # Pause pour éviter de surcharger l'API
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du bot: {e}")
            self.notifier.send_notification("Erreur de bot de trading", f"Une erreur s'est produite: {e}")
            
        finally:
            # Nettoyage
            self._cleanup()
    
    def stop(self):
        """
        Arrête le bot de trading
        """
        if not self.running:
            logger.warning("Le bot n'est pas en cours d'exécution")
            return
            
        logger.info("Arrêt du bot de trading")
        self.running = False
        
        # Attendre que le bot s'arrête proprement
        time.sleep(1)
        
        # Afficher les statistiques
        self._print_stats()
    
    def _check_and_execute(self, symbol):
        """
        Vérifie les signaux et exécute les trades pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
        """
        # Vérifier si nous devons mettre à jour les données
        current_time = datetime.now()
        
        # Si c'est la première vérification ou si le temps est écoulé
        if symbol not in self.last_check_time or \
           (current_time - self.last_check_time[symbol]).total_seconds() >= self._get_check_interval(symbol):
            
            try:
                # Récupérer les données historiques
                timeframe = TRADING.get('timeframe', '15m')
                data = self._fetch_data(symbol, timeframe)
                
                # Ajouter les indicateurs techniques
                data = TechnicalIndicators.add_all_indicators(data)
                
                # Générer les signaux
                signals = self.strategy.run(data)
                
                # Obtenir le dernier signal
                last_signal = signals.iloc[-1] if not signals.empty else 0
                
                # Exécuter le signal
                if last_signal != 0:
                    result = self.trader.execute_signal(symbol, last_signal, data)
                    
                    if result:
                        action = result.get('action')
                        if action == 'open':
                            side = result.get('side')
                            price = result.get('price')
                            size = result.get('size')
                            stop_loss = result.get('stop_loss')
                            take_profit = result.get('take_profit')
                            
                            message = (f"Position ouverte: {side} {size} {symbol} @ {price}\n"
                                      f"Stop Loss: {stop_loss}\nTake Profit: {take_profit}")
                            
                            logger.info(message)
                            self.notifier.send_notification(f"Position ouverte sur {symbol}", message)
                            
                        elif action == 'close':
                            side = result.get('side')
                            entry_price = result.get('entry_price')
                            exit_price = result.get('exit_price')
                            size = result.get('size')
                            pnl = result.get('pnl')
                            reason = result.get('reason')
                            
                            message = (f"Position fermée: {side} {size} {symbol}\n"
                                      f"Entrée: {entry_price}, Sortie: {exit_price}\n"
                                      f"P/L: {pnl:.2f} ({pnl / (entry_price * size) * 100:.2f}%)\n"
                                      f"Raison: {reason}")
                            
                            logger.info(message)
                            self.notifier.send_notification(f"Position fermée sur {symbol}", message)
                
                # Mettre à jour le temps de dernière vérification
                self.last_check_time[symbol] = current_time
                
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de {symbol}: {e}")
    
    def _update_trailing_stops(self):
        """
        Met à jour les trailing stops pour toutes les positions actives
        """
        try:
            # Récupérer les positions actives
            positions = self.trader.get_positions()
            
            for symbol, position in list(positions.items()):
                # Récupérer le prix actuel
                current_price = self._get_current_price(symbol)
                
                if current_price:
                    # Mettre à jour le trailing stop
                    self.trader.update_trailing_stops(symbol, current_price)
                    
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des trailing stops: {e}")
    
    def _fetch_data(self, symbol, timeframe):
        """
        Récupère les données historiques
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            timeframe (str): Timeframe (ex: 15m, 1h, 4h)
            
        Returns:
            pd.DataFrame: Données historiques
        """
        # Déterminer le nombre de bougies à récupérer
        limit = 500  # Par défaut
        
        # Récupérer les données
        data = self.fetcher.fetch_historical_ohlcv(symbol, timeframe, limit=limit)
        
        # Si pas de données, renvoyer un DataFrame vide
        if not data or len(data) == 0:
            logger.warning(f"Pas de données pour {symbol} {timeframe}")
            return pd.DataFrame()
            
        # Convertir en DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            df = data
            
        return df
    
    def _get_check_interval(self, symbol):
        """
        Détermine l'intervalle entre les vérifications
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            int: Intervalle en secondes
        """
        timeframe = TRADING.get('timeframe', '15m')
        
        # Convertir le timeframe en secondes
        if timeframe.endswith('m'):
            interval = int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            interval = int(timeframe[:-1]) * 60 * 60
        elif timeframe.endswith('d'):
            interval = int(timeframe[:-1]) * 60 * 60 * 24
        else:
            # Valeur par défaut: 15 minutes
            interval = 15 * 60
            
        # Réduire l'intervalle pour être plus réactif (1/4 du timeframe)
        interval = max(30, interval // 4)
        
        return interval
    
    def _get_current_price(self, symbol):
        """
        Récupère le prix actuel d'un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            float: Prix actuel
        """
        try:
            ticker = self.exchange.get_ticker(symbol)
            if ticker:
                return ticker['last']
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix de {symbol}: {e}")
            
        return None
    
    def _cleanup(self):
        """
        Nettoie les ressources avant l'arrêt
        """
        logger.info("Nettoyage des ressources")
        
        # Fermer toutes les positions en mode paper
        if self.mode == "paper":
            positions = self.trader.get_positions()
            for symbol in list(positions.keys()):
                current_price = self._get_current_price(symbol)
                if current_price:
                    self.trader.execute_signal(symbol, 0, None)  # Signal 0 pour fermer
    
    def _print_stats(self):
        """
        Affiche les statistiques du bot
        """
        stats = self.trader.get_stats()
        
        logger.info("Statistiques du bot:")
        logger.info(f"Capital initial: {stats.get('initial_capital', 0)}")
        logger.info(f"Capital final: {stats.get('current_capital', 0)}")
        logger.info(f"Profit/Perte: {stats.get('profit_loss', 0)} ({stats.get('profit_loss_pct', 0)}%)")
        logger.info(f"Nombre de trades: {stats.get('total_trades', 0)}")
        logger.info(f"Trades gagnants: {stats.get('winning_trades', 0)}")
        logger.info(f"Trades perdants: {stats.get('losing_trades', 0)}")
        logger.info(f"Taux de réussite: {stats.get('win_rate', 0)}%")
        logger.info(f"Drawdown maximum: {stats.get('max_drawdown', 0)}%")
        
        # Envoyer un rapport par email
        self.notifier.send_notification(
            "Rapport du bot de trading",
            f"Capital initial: {stats.get('initial_capital', 0)}\n"
            f"Capital final: {stats.get('current_capital', 0)}\n"
            f"Profit/Perte: {stats.get('profit_loss', 0)} ({stats.get('profit_loss_pct', 0)}%)\n"
            f"Nombre de trades: {stats.get('total_trades', 0)}\n"
            f"Taux de réussite: {stats.get('win_rate', 0)}%\n"
            f"Drawdown maximum: {stats.get('max_drawdown', 0)}%"
        )


# Point d'entrée principal
if __name__ == "__main__":
    try:
        # Initialiser le bot
        bot = TradingBot()
        
        # Démarrer le bot
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Interruption clavier, arrêt du bot")
        
    except Exception as e:
        logger.error(f"Erreur non gérée: {e}")
        
    finally:
        logger.info("Fin du programme")