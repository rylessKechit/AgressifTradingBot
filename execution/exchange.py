"""
Module d'interface avec les exchanges
Gère la connexion et les interactions avec les plateformes d'échange
"""
import time
import logging
import ccxt
from config.settings import EXCHANGE as exchange_config

logger = logging.getLogger(__name__)

class Exchange:
    """
    Classe qui gère l'interface avec les plateformes d'échange
    """
    
    def __init__(self, exchange_id=None, api_key=None, api_secret=None, testnet=None):
        """
        Initialise la connexion à l'exchange
        
        Args:
            exchange_id (str, optional): ID de l'exchange (binance, ftx, kraken, etc.)
            api_key (str, optional): Clé API
            api_secret (str, optional): Secret API
            testnet (bool, optional): Utiliser le testnet/sandbox
        """
        # Utiliser les paramètres passés ou ceux de la configuration
        self.exchange_id = exchange_id or exchange_config['name']
        self.api_key = api_key or exchange_config['api_key']
        self.api_secret = api_secret or exchange_config['api_secret']
        self.testnet = testnet if testnet is not None else exchange_config['testnet']
        
        # Initialiser l'exchange
        self.exchange = self._initialize_exchange()
        
        # Vérifier la connexion
        self._check_connection()
    
    def _initialize_exchange(self):
        """
        Initialise la connexion à l'exchange avec les paramètres appropriés
        
        Returns:
            ccxt.Exchange: Instance de l'exchange initialisé
        """
        try:
            # Récupérer la classe de l'exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Paramètres
            params = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            }
            
            # Ajouter les options spécifiques à l'exchange
            if self.exchange_id == 'binance' and self.testnet:
                params['options'] = {'defaultType': 'future'}
                params['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
            elif self.exchange_id == 'ftx' and self.testnet:
                # FTX a son propre environnement de test
                params['urls'] = {'api': 'https://ftx.com/api'}
            
            # Créer l'instance de l'exchange
            exchange = exchange_class(params)
            
            logger.info(f"Exchange {self.exchange_id} initialisé avec succès")
            return exchange
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange {self.exchange_id}: {e}")
            raise
    
    def _check_connection(self):
        """
        Vérifie que la connexion à l'exchange fonctionne
        
        Returns:
            bool: True si la connexion est établie, False sinon
        """
        try:
            # Tester la connexion en récupérant les tickers
            self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"Connexion à {self.exchange_id} établie avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion à {self.exchange_id}: {e}")
            return False
    
    def get_balance(self):
        """
        Récupère le solde du compte
        
        Returns:
            dict: Solde du compte
        """
        try:
            balance = self.exchange.fetch_balance()
            logger.info(f"Solde récupéré: {balance['total']}")
            return balance
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {e}")
            return None
    
    def create_market_order(self, symbol, side, amount):
        """
        Crée un ordre au marché
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de l'ordre ('buy' ou 'sell')
            amount (float): Quantité à acheter/vendre
            
        Returns:
            dict: Informations sur l'ordre créé
        """
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Ordre au marché créé: {side} {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre au marché: {e}")
            return None
    
    def create_limit_order(self, symbol, side, amount, price):
        """
        Crée un ordre à cours limité
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de l'ordre ('buy' ou 'sell')
            amount (float): Quantité à acheter/vendre
            price (float): Prix limite
            
        Returns:
            dict: Informations sur l'ordre créé
        """
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"Ordre à cours limité créé: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre à cours limité: {e}")
            return None
    
    def create_stop_loss_order(self, symbol, side, amount, price):
        """
        Crée un ordre stop-loss
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de l'ordre ('buy' ou 'sell')
            amount (float): Quantité à acheter/vendre
            price (float): Prix d'activation du stop-loss
            
        Returns:
            dict: Informations sur l'ordre créé
        """
        try:
            params = {'stopPrice': price}
            order_type = 'stop_market'
            
            # Certains exchanges utilisent des paramètres différents
            if self.exchange_id == 'binance':
                order_type = 'STOP_MARKET'
            elif self.exchange_id == 'ftx':
                params = {'stopPrice': price, 'reduceOnly': True}
            
            order = self.exchange.create_order(symbol, order_type, side, amount, None, params)
            logger.info(f"Ordre stop-loss créé: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre stop-loss: {e}")
            return None
    
    def create_take_profit_order(self, symbol, side, amount, price):
        """
        Crée un ordre take-profit
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de l'ordre ('buy' ou 'sell')
            amount (float): Quantité à acheter/vendre
            price (float): Prix d'activation du take-profit
            
        Returns:
            dict: Informations sur l'ordre créé
        """
        try:
            params = {'stopPrice': price}
            order_type = 'take_profit_market'
            
            # Certains exchanges utilisent des paramètres différents
            if self.exchange_id == 'binance':
                order_type = 'TAKE_PROFIT_MARKET'
            elif self.exchange_id == 'ftx':
                params = {'triggerPrice': price, 'reduceOnly': True}
                order_type = 'take_profit'
            
            order = self.exchange.create_order(symbol, order_type, side, amount, None, params)
            logger.info(f"Ordre take-profit créé: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre take-profit: {e}")
            return None
    
    def cancel_order(self, order_id, symbol):
        """
        Annule un ordre
        
        Args:
            order_id (str): ID de l'ordre à annuler
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            dict: Informations sur l'ordre annulé
        """
        try:
            canceled_order = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Ordre {order_id} sur {symbol} annulé")
            return canceled_order
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre {order_id} sur {symbol}: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """
        Récupère les ordres ouverts
        
        Args:
            symbol (str, optional): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            list: Liste des ordres ouverts
        """
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            logger.info(f"Ordres ouverts récupérés: {len(open_orders)}")
            return open_orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {e}")
            return []
    
    def get_closed_orders(self, symbol=None, since=None, limit=None):
        """
        Récupère les ordres fermés
        
        Args:
            symbol (str, optional): Symbole de la paire (ex: BTC/USDT)
            since (int, optional): Timestamp de début en millisecondes
            limit (int, optional): Nombre maximum d'ordres à récupérer
            
        Returns:
            list: Liste des ordres fermés
        """
        try:
            closed_orders = self.exchange.fetch_closed_orders(symbol, since, limit)
            logger.info(f"Ordres fermés récupérés: {len(closed_orders)}")
            return closed_orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres fermés: {e}")
            return []
    
    def get_order_status(self, order_id, symbol):
        """
        Récupère le statut d'un ordre
        
        Args:
            order_id (str): ID de l'ordre
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            dict: Informations sur l'ordre
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            logger.info(f"Statut de l'ordre {order_id} sur {symbol}: {order['status']}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut de l'ordre {order_id} sur {symbol}: {e}")
            return None
    
    def get_positions(self):
        """
        Récupère les positions ouvertes (pour les futures)
        
        Returns:
            list: Liste des positions ouvertes
        """
        try:
            if not hasattr(self.exchange, 'fetch_positions'):
                logger.warning(f"L'exchange {self.exchange_id} ne supporte pas la récupération des positions")
                return []
                
            positions = self.exchange.fetch_positions()
            logger.info(f"Positions récupérées: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions: {e}")
            return []
    
    def get_ticker(self, symbol):
        """
        Récupère le ticker pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            dict: Ticker
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker pour {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol, timeframe='1h', limit=100):
        """
        Récupère les données OHLCV pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            timeframe (str, optional): Timeframe (ex: 1m, 5m, 15m, 1h, 4h, 1d)
            limit (int, optional): Nombre maximum de bougies
            
        Returns:
            list: Liste des bougies OHLCV
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV pour {symbol}: {e}")
            return []
    
    def get_order_book(self, symbol, limit=20):
        """
        Récupère le carnet d'ordres pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            limit (int, optional): Profondeur du carnet d'ordres
            
        Returns:
            dict: Carnet d'ordres
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres pour {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol, capital, risk_percent=1.0, stop_loss_percent=2.0):
        """
        Calcule la taille de position optimale selon la gestion des risques
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            capital (float): Capital total
            risk_percent (float, optional): Pourcentage du capital à risquer
            stop_loss_percent (float, optional): Pourcentage du stop loss
            
        Returns:
            float: Taille de la position
        """
        try:
            # Obtenir le ticker actuel
            ticker = self.get_ticker(symbol)
            
            if not ticker:
                logger.error(f"Impossible de calculer la taille de position pour {symbol}: ticker non disponible")
                return 0
            
            current_price = ticker['last']
            
            # Montant à risquer
            risk_amount = capital * (risk_percent / 100)
            
            # Calcul de la taille de position basée sur le stop loss
            position_size = risk_amount / (current_price * (stop_loss_percent / 100))
            
            # Arrondir la taille de position selon les règles de l'exchange
            if hasattr(self.exchange, 'amount_to_precision'):
                position_size = self.exchange.amount_to_precision(symbol, position_size)
            else:
                # Arrondi par défaut
                position_size = round(position_size, 6)
            
            logger.info(f"Taille de position calculée pour {symbol}: {position_size}")
            return float(position_size)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position pour {symbol}: {e}")
            return 0
    
    def get_min_order_amount(self, symbol):
        """
        Récupère le montant minimum pour un ordre
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            float: Montant minimum
        """
        try:
            # Récupérer les informations sur le marché
            market = self.exchange.market(symbol)
            
            if 'limits' in market and 'amount' in market['limits'] and 'min' in market['limits']['amount']:
                min_amount = market['limits']['amount']['min']
                logger.info(f"Montant minimum pour {symbol}: {min_amount}")
                return min_amount
            else:
                logger.warning(f"Impossible de trouver le montant minimum pour {symbol}, utilisation de la valeur par défaut")
                return 0.001  # Valeur par défaut pour BTC
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du montant minimum pour {symbol}: {e}")
            return 0.001  # Valeur par défaut
    
    def get_price_precision(self, symbol):
        """
        Récupère la précision de prix pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            int: Nombre de décimales pour le prix
        """
        try:
            # Récupérer les informations sur le marché
            market = self.exchange.market(symbol)
            
            if 'precision' in market and 'price' in market['precision']:
                precision = market['precision']['price']
                return precision
            else:
                logger.warning(f"Impossible de trouver la précision de prix pour {symbol}, utilisation de la valeur par défaut")
                return 2  # Valeur par défaut
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la précision de prix pour {symbol}: {e}")
            return 2  # Valeur par défaut


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialiser l'exchange
    exchange = Exchange()
    
    # Récupérer le solde
    balance = exchange.get_balance()
    
    if balance:
        print("Solde total:")
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"{currency}: {amount}")
    
    # Récupérer le ticker pour BTC/USDT
    ticker = exchange.get_ticker("BTC/USDT")
    
    if ticker:
        print(f"\nTicker BTC/USDT:")
        print(f"Prix: {ticker['last']}")
        print(f"Bid: {ticker['bid']}")
        print(f"Ask: {ticker['ask']}")
        print(f"Volume: {ticker['volume']}")
        
    # Calculer la taille de position
    position_size = exchange.calculate_position_size("BTC/USDT", 10000, 1.0, 2.0)
    print(f"\nTaille de position pour BTC/USDT: {position_size}")