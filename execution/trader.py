"""
Module d'exécution des trades
Gère l'exécution des ordres en fonction des signaux générés par les stratégies
"""
import logging
import time
from datetime import datetime
import pandas as pd
from execution.exchange import Exchange
from execution.order import Order
from risk.position_sizing import PositionSizer
from config.settings import CAPITAL, TRADING

logger = logging.getLogger(__name__)

class Trader:
    """
    Classe qui gère l'exécution des trades
    """
    
    def __init__(self, exchange=None, position_sizer=None, mode="backtest"):
        """
        Initialise le Trader
        
        Args:
            exchange (Exchange, optional): Instance de l'exchange
            position_sizer (PositionSizer, optional): Instance du calculateur de taille de position
            mode (str, optional): Mode d'exécution ("backtest", "paper", "live")
        """
        self.exchange = exchange or Exchange()
        self.position_sizer = position_sizer or PositionSizer()
        self.mode = mode
        
        # Positions et ordres actifs
        self.active_positions = {}  # {symbol: {side, entry_price, size, entry_time, stop_loss, take_profit}}
        self.active_orders = {}  # {order_id: {symbol, side, type, price, amount, status}}
        
        # Historique des trades
        self.trades_history = []
        
        # Capital initial et actuel
        self.initial_capital = CAPITAL.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        
        # Statistiques
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'current_drawdown': 0
        }
        
        logger.info(f"Trader initialisé en mode {mode} avec un capital initial de {self.initial_capital}")
    
    def execute_signal(self, symbol, signal, data=None, risk_params=None):
        """
        Exécute un signal de trading
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            signal (int): Signal (-1=vente, 0=neutre, 1=achat)
            data (pd.DataFrame, optional): Données de marché
            risk_params (dict, optional): Paramètres de gestion des risques
            
        Returns:
            dict: Informations sur l'exécution
        """
        if signal == 0:
            logger.debug(f"Pas de signal pour {symbol}, aucune action")
            return None
        
        # Récupérer les paramètres de risque
        risk_params = risk_params or {}
        risk_per_trade = risk_params.get('risk_per_trade', CAPITAL.get('risk_per_trade', 0.01))
        max_positions = risk_params.get('max_positions', CAPITAL.get('max_positions', 5))
        
        # Vérifier si nous avons déjà atteint le nombre maximum de positions
        if len(self.active_positions) >= max_positions and symbol not in self.active_positions:
            logger.warning(f"Nombre maximum de positions atteint ({max_positions}), signal ignoré pour {symbol}")
            return None
        
        # Récupérer le prix actuel
        current_price = self._get_current_price(symbol, data)
        
        if current_price is None:
            logger.error(f"Impossible d'obtenir le prix actuel pour {symbol}")
            return None
        
        # Vérifier si nous avons déjà une position sur ce symbole
        if symbol in self.active_positions:
            return self._handle_existing_position(symbol, signal, current_price, data)
        else:
            return self._open_new_position(symbol, signal, current_price, data, risk_per_trade)
    
    def _get_current_price(self, symbol, data=None):
        """
        Récupère le prix actuel d'un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            data (pd.DataFrame, optional): Données de marché
            
        Returns:
            float: Prix actuel
        """
        if self.mode == "backtest" and data is not None:
            # En mode backtest, utiliser les données fournies
            return data['close'].iloc[-1]
        else:
            # En mode paper ou live, interroger l'exchange
            ticker = self.exchange.get_ticker(symbol)
            if ticker:
                return ticker['last']
            return None
    
    def _handle_existing_position(self, symbol, signal, current_price, data=None):
        """
        Gère une position existante
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            signal (int): Signal (-1=vente, 0=neutre, 1=achat)
            current_price (float): Prix actuel
            data (pd.DataFrame, optional): Données de marché
            
        Returns:
            dict: Informations sur l'exécution
        """
        position = self.active_positions[symbol]
        
        # Si le signal est opposé à la position actuelle, fermer la position
        if (position['side'] == 'long' and signal == -1) or (position['side'] == 'short' and signal == 1):
            return self._close_position(symbol, current_price, f"Signal {signal}")
        
        # Vérifier le stop loss
        if position['side'] == 'long' and current_price <= position['stop_loss']:
            return self._close_position(symbol, current_price, "Stop Loss")
        elif position['side'] == 'short' and current_price >= position['stop_loss']:
            return self._close_position(symbol, current_price, "Stop Loss")
        
        # Vérifier le take profit
        if position['side'] == 'long' and current_price >= position['take_profit']:
            return self._close_position(symbol, current_price, "Take Profit")
        elif position['side'] == 'short' and current_price <= position['take_profit']:
            return self._close_position(symbol, current_price, "Take Profit")
        
        # Si nous sommes ici, nous n'avons rien fait
        return None
    
    def _open_new_position(self, symbol, signal, current_price, data=None, risk_per_trade=0.01):
        """
        Ouvre une nouvelle position
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            signal (int): Signal (-1=vente, 0=neutre, 1=achat)
            current_price (float): Prix actuel
            data (pd.DataFrame, optional): Données de marché
            risk_per_trade (float, optional): Pourcentage du capital à risquer par trade
            
        Returns:
            dict: Informations sur l'exécution
        """
        # Déterminer le côté (long/short)
        side = 'long' if signal == 1 else 'short'
        
        # Calculer le stop loss
        stop_loss = self._calculate_stop_loss(symbol, side, current_price, data)
        
        # Calculer le take profit
        take_profit = self._calculate_take_profit(current_price, stop_loss, side)
        
        # Calculer la taille de la position
        size = self.position_sizer.calculate_position_size(
            self.current_capital,
            current_price,
            abs(current_price - stop_loss) / current_price,
            risk_per_trade
        )
        
        if size <= 0:
            logger.warning(f"Taille de position calculée trop petite pour {symbol}, ordre ignoré")
            return None
        
        # En mode live ou paper, passer l'ordre
        order_result = None
        if self.mode in ["live", "paper"]:
            order_side = 'buy' if side == 'long' else 'sell'
            order_result = self.exchange.create_market_order(symbol, order_side, size)
            
            # Création des ordres stop loss et take profit
            if order_result and self.mode == "live":
                sl_side = 'sell' if side == 'long' else 'buy'
                tp_side = 'sell' if side == 'long' else 'buy'
                
                self.exchange.create_stop_loss_order(symbol, sl_side, size, stop_loss)
                self.exchange.create_take_profit_order(symbol, tp_side, size, take_profit)
        
        # En mode backtest, simuler l'exécution de l'ordre
        else:
            order_result = {
                'id': f"backtest-{len(self.trades_history)}",
                'symbol': symbol,
                'side': 'buy' if side == 'long' else 'sell',
                'price': current_price,
                'amount': size,
                'timestamp': int(time.time() * 1000)
            }
        
        # Enregistrer la position
        self.active_positions[symbol] = {
            'side': side,
            'entry_price': current_price,
            'size': size,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'order_id': order_result['id'] if order_result else None
        }
        
        logger.info(f"Position ouverte: {side} {size} {symbol} @ {current_price} (SL: {stop_loss}, TP: {take_profit})")
        
        return {
            'action': 'open',
            'symbol': symbol,
            'side': side,
            'price': current_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'order_result': order_result
        }
    
    def _close_position(self, symbol, current_price, reason):
        """
        Ferme une position existante
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            current_price (float): Prix actuel
            reason (str): Raison de la fermeture
            
        Returns:
            dict: Informations sur l'exécution
        """
        if symbol not in self.active_positions:
            logger.warning(f"Pas de position active sur {symbol} à fermer")
            return None
        
        position = self.active_positions[symbol]
        
        # Calculer le P/L
        if position['side'] == 'long':
            pnl = (current_price - position['entry_price']) / position['entry_price'] * position['size'] * position['entry_price']
        else:  # short
            pnl = (position['entry_price'] - current_price) / position['entry_price'] * position['size'] * position['entry_price']
        
        # En mode live ou paper, passer l'ordre
        order_result = None
        if self.mode in ["live", "paper"]:
            order_side = 'sell' if position['side'] == 'long' else 'buy'
            order_result = self.exchange.create_market_order(symbol, order_side, position['size'])
            
            # Annuler les ordres stop loss et take profit en mode live
            if self.mode == "live":
                # Récupérer et annuler les ordres ouverts
                open_orders = self.exchange.get_open_orders(symbol)
                for order in open_orders:
                    self.exchange.cancel_order(order['id'], symbol)
        
        # En mode backtest, simuler l'exécution de l'ordre
        else:
            order_result = {
                'id': f"backtest-close-{len(self.trades_history)}",
                'symbol': symbol,
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'price': current_price,
                'amount': position['size'],
                'timestamp': int(time.time() * 1000)
            }
        
        # Mettre à jour le capital
        self.current_capital += pnl
        
        # Enregistrer le trade dans l'historique
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'pnl': pnl,
            'pnl_percent': (pnl / (position['entry_price'] * position['size'])) * 100,
            'reason': reason
        }
        
        self.trades_history.append(trade)
        
        # Mettre à jour les statistiques
        self.stats['total_trades'] += 1
        if pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        self.stats['total_profit'] += pnl
        
        # Calculer le drawdown
        peak_capital = max(self.initial_capital, self.current_capital)
        current_drawdown = (peak_capital - self.current_capital) / peak_capital * 100
        
        if current_drawdown > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = current_drawdown
        
        self.stats['current_drawdown'] = current_drawdown
        
        # Supprimer la position
        del self.active_positions[symbol]
        
        logger.info(f"Position fermée: {position['side']} {position['size']} {symbol} @ {current_price}, P/L: {pnl:.2f} ({reason})")
        
        return {
            'action': 'close',
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'pnl': pnl,
            'reason': reason,
            'order_result': order_result
        }
    
    def _calculate_stop_loss(self, symbol, side, current_price, data=None):
        """
        Calcule le niveau de stop loss
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de la position ('long' ou 'short')
            current_price (float): Prix actuel
            data (pd.DataFrame, optional): Données de marché
            
        Returns:
            float: Niveau de stop loss
        """
        # Pourcentage de stop loss par défaut
        default_stop_pct = 0.03  # 3%
        
        # Si nous avons des données, essayer de calculer un stop dynamique basé sur ATR
        if data is not None and 'atr_14' in data.columns:
            atr = data['atr_14'].iloc[-1]
            atr_multiplier = 2.0
            
            if side == 'long':
                stop_loss = current_price - (atr * atr_multiplier)
            else:  # short
                stop_loss = current_price + (atr * atr_multiplier)
                
            return stop_loss
        
        # Sinon, utiliser un stop fixe basé sur un pourcentage
        if side == 'long':
            stop_loss = current_price * (1 - default_stop_pct)
        else:  # short
            stop_loss = current_price * (1 + default_stop_pct)
            
        return stop_loss
    
    def _calculate_take_profit(self, current_price, stop_loss, side, risk_reward_ratio=2.0):
        """
        Calcule le niveau de take profit
        
        Args:
            current_price (float): Prix actuel
            stop_loss (float): Niveau de stop loss
            side (str): Côté de la position ('long' ou 'short')
            risk_reward_ratio (float, optional): Ratio risque/récompense
            
        Returns:
            float: Niveau de take profit
        """
        risk = abs(current_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if side == 'long':
            take_profit = current_price + reward
        else:  # short
            take_profit = current_price - reward
            
        return take_profit
    
    def update_trailing_stops(self, symbol, current_price):
        """
        Met à jour les stop loss trailing
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            current_price (float): Prix actuel
            
        Returns:
            bool: True si le stop a été modifié, False sinon
        """
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        
        # Paramètres du trailing stop
        trailing_activation_pct = 0.015  # 1.5% de profit pour activer
        trailing_distance_pct = 0.01  # 1% de distance
        
        # Calculer le profit actuel
        if position['side'] == 'long':
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Si le profit est supérieur au seuil d'activation
            if profit_pct >= trailing_activation_pct:
                # Calculer le nouveau stop loss
                new_stop = current_price * (1 - trailing_distance_pct)
                
                # Si le nouveau stop est supérieur à l'ancien, le mettre à jour
                if new_stop > position['stop_loss']:
                    old_stop = position['stop_loss']
                    position['stop_loss'] = new_stop
                    
                    logger.info(f"Trailing stop mis à jour pour {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    
                    # En mode live, mettre à jour l'ordre stop loss
                    if self.mode == "live":
                        # Récupérer et annuler l'ancien ordre stop loss
                        open_orders = self.exchange.get_open_orders(symbol)
                        for order in open_orders:
                            if order['type'] == 'stop_market' or order['type'] == 'stop':
                                self.exchange.cancel_order(order['id'], symbol)
                                
                        # Créer un nouvel ordre stop loss
                        self.exchange.create_stop_loss_order(symbol, 'sell', position['size'], new_stop)
                    
                    return True
                    
        elif position['side'] == 'short':
            profit_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # Si le profit est supérieur au seuil d'activation
            if profit_pct >= trailing_activation_pct:
                # Calculer le nouveau stop loss
                new_stop = current_price * (1 + trailing_distance_pct)
                
                # Si le nouveau stop est inférieur à l'ancien, le mettre à jour
                if new_stop < position['stop_loss']:
                    old_stop = position['stop_loss']
                    position['stop_loss'] = new_stop
                    
                    logger.info(f"Trailing stop mis à jour pour {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    
                    # En mode live, mettre à jour l'ordre stop loss
                    if self.mode == "live":
                        # Récupérer et annuler l'ancien ordre stop loss
                        open_orders = self.exchange.get_open_orders(symbol)
                        for order in open_orders:
                            if order['type'] == 'stop_market' or order['type'] == 'stop':
                                self.exchange.cancel_order(order['id'], symbol)
                                
                        # Créer un nouvel ordre stop loss
                        self.exchange.create_stop_loss_order(symbol, 'buy', position['size'], new_stop)
                    
                    return True
        
        return False
    
    def get_stats(self):
        """
        Récupère les statistiques de trading
        
        Returns:
            dict: Statistiques de trading
        """
        # Calculer des statistiques supplémentaires
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
        else:
            self.stats['win_rate'] = 0
            
        self.stats['current_capital'] = self.current_capital
        self.stats['profit_loss'] = self.current_capital - self.initial_capital
        self.stats['profit_loss_pct'] = (self.stats['profit_loss'] / self.initial_capital) * 100
        
        return self.stats
    
    def get_position(self, symbol):
        """
        Récupère une position active
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            dict: Position active ou None
        """
        return self.active_positions.get(symbol)
    
    def get_positions(self):
        """
        Récupère toutes les positions actives
        
        Returns:
            dict: Positions actives
        """
        return self.active_positions
    
    def get_trades_history(self):
        """
        Récupère l'historique des trades
        
        Returns:
            list: Historique des trades
        """
        return self.trades_history


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    from data.fetcher import DataFetcher
    from data.indicators import TechnicalIndicators
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialiser le trader en mode backtest
    trader = Trader(mode="backtest")
    
    # Récupérer des données pour le test
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_ohlcv("BTC/USDT", "1h", limit=100)
    
    # Convertir en DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Ajouter les indicateurs
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Générer des signaux de test
    signals = pd.Series(0, index=df.index)
    signals.iloc[20] = 1  # Signal d'achat
    signals.iloc[50] = -1  # Signal de vente
    
    # Simuler l'exécution des signaux
    for i in range(len(df)):
        signal = signals.iloc[i]
        data_slice = df.iloc[:i+1]
        current_price = df['close'].iloc[i]
        
        # Exécuter le signal
        if signal != 0:
            result = trader.execute_signal("BTC/USDT", signal, data_slice)
            print(f"Exécution: {result}")
        
        # Mettre à jour les trailing stops
        for symbol in list(trader.active_positions.keys()):
            trader.update_trailing_stops(symbol, current_price)
    
    # Afficher les statistiques
    stats = trader.get_stats()
    print("\nStatistiques:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Afficher l'historique des trades
    print("\nHistorique des trades:")
    for trade in trader.get_trades_history():
        print(f"{trade['symbol']} {trade['side']} - Entrée: {trade['entry_price']:.2f}, Sortie: {trade['exit_price']:.2f}, P/L: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%), Raison: {trade['reason']}")