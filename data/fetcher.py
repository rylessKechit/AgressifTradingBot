"""
Module de récupération des données de marché
Gère la récupération des données historiques et en temps réel depuis différentes sources
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
import logging

from config.settings import DATA_DIR, EXCHANGE

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Classe pour récupérer les données de marché depuis différentes sources
    """
    
    def __init__(self, exchange_id=None, use_testnet=None):
        """
        Initialise le DataFetcher avec la configuration de l'exchange
        
        Args:
            exchange_id (str, optional): ID de l'exchange (binance, ftx, kraken, etc.)
            use_testnet (bool, optional): Utiliser le testnet/sandbox
        """
        # Utiliser les paramètres passés ou ceux de la configuration
        self.exchange_id = exchange_id or EXCHANGE["name"]
        self.use_testnet = use_testnet if use_testnet is not None else EXCHANGE["testnet"]
        
        # Initialiser l'exchange
        self.exchange = self._initialize_exchange()
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(os.path.join(DATA_DIR, self.exchange_id), exist_ok=True)
    
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
                'apiKey': EXCHANGE['api_key'],
                'secret': EXCHANGE['api_secret'],
                'enableRateLimit': True,
            }
            
            # Ajouter les options spécifiques à l'exchange
            if self.exchange_id == 'binance' and self.use_testnet:
                params['options'] = {'defaultType': 'future'}
                params['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
            elif self.exchange_id == 'ftx' and self.use_testnet:
                # FTX a son propre environnement de test
                params['urls'] = {'api': 'https://ftx.com/api'}
            
            # Créer l'instance de l'exchange
            exchange = exchange_class(params)
            
            logger.info(f"Exchange {self.exchange_id} initialisé avec succès")
            return exchange
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange {self.exchange_id}: {e}")
            raise
    
    def fetch_historical_ohlcv(self, symbol, timeframe, start_date=None, end_date=None, limit=1000, 
                          save_to_file=True, use_cache=True):
        """
        Récupère les données OHLCV historiques depuis l'exchange
        """
        # Valeurs par défaut pour les dates
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Récupération des données pour {symbol} sur {timeframe} de {start_date} à {end_date}")
        
        # Vérification du cache...
        
        # Convertir les dates en timestamps
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        logger.info(f"Récupération des données entre {since} et {until} ms")
        
        # Récupérer les données par lots
        all_ohlcv = []
        current_since = since
        
        while current_since < until:
            try:
                logger.info(f"Récupération depuis {datetime.fromtimestamp(current_since/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Vérifier si l'exchange est initialisé
                if not self.exchange:
                    logger.error("Exchange non initialisé")
                    break
                    
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not ohlcv or len(ohlcv) == 0:
                    logger.warning(f"Aucune donnée récupérée depuis {datetime.fromtimestamp(current_since/1000)}")
                    break
                    
                logger.info(f"Récupéré {len(ohlcv)} bougies depuis {datetime.fromtimestamp(current_since/1000)}")
                all_ohlcv.extend(ohlcv)
                
                # Mise à jour du timestamp pour la prochaine requête
                current_since = ohlcv[-1][0] + 1
                
                # Respecter les limites de l'API
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données: {e}")
                # Attendre un peu plus longtemps en cas d'erreur
                time.sleep(5)
                # Réessayer avec un timestamp plus récent
                current_since += 60 * 60 * 1000  # +1 heure
        
        # Convertir en DataFrame
        if not all_ohlcv:
            logger.warning(f"Aucune donnée récupérée pour {symbol} {timeframe}")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Enlever les doublons potentiels
        df = df[~df.index.duplicated(keep='first')]
        
        # Afficher des statistiques sur les données
        logger.info(f"Données récupérées: {len(df)} bougies de {df.index.min()} à {df.index.max()}")
        logger.info(f"Plage de prix: {df['low'].min()} - {df['high'].max()}")
        logger.info(f"Volume total: {df['volume'].sum()}")
        
        # Vérifier la présence de NaN
        if df.isnull().any().any():
            logger.warning("Les données contiennent des valeurs NaN")
            # Compter les valeurs NaN par colonne
            for col in df.columns:
                nan_count = df[col].isnull().sum()
                if nan_count > 0:
                    logger.warning(f"Colonne {col}: {nan_count} valeurs NaN")
        
        return df

    def fetch_ticker(self, symbol):
        """
        Récupère le ticker actuel pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
        
        Returns:
            dict: Ticker actuel
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker pour {symbol}: {e}")
            return None
    
    def fetch_order_book(self, symbol, limit=20):
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
    
    def fetch_trades(self, symbol, since=None, limit=100):
        """
        Récupère les transactions récentes pour un symbole
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            since (int, optional): Timestamp de début en millisecondes
            limit (int, optional): Nombre maximum de transactions
        
        Returns:
            list: Liste des transactions récentes
        """
        try:
            trades = self.exchange.fetch_trades(symbol, since, limit)
            return trades
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des transactions pour {symbol}: {e}")
            return []
    
    def fetch_multi_timeframe_data(self, symbol, timeframes, start_date=None, end_date=None):
        """
        Récupère les données pour plusieurs timeframes
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            timeframes (list): Liste des timeframes (ex: ['15m', '1h', '4h'])
            start_date (datetime, optional): Date de début
            end_date (datetime, optional): Date de fin
        
        Returns:
            dict: Dictionnaire avec les DataFrames pour chaque timeframe
        """
        result = {}
        
        for tf in timeframes:
            df = self.fetch_historical_ohlcv(symbol, tf, start_date, end_date)
            result[tf] = df
            
        return result


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test de récupération de données
    fetcher = DataFetcher()
    
    # Récupérer les données historiques
    df = fetcher.fetch_historical_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    print(f"Données récupérées: {len(df)} lignes")
    print(df.head())