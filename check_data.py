# check_data.py
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def main():
    """Vérification des données et indicateurs"""
    # Récupérer des données
    fetcher = DataFetcher()
    
    symbol = "BTC/USDT"
    timeframe = "15m"  # Utiliser un timeframe plus court
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 1, 15)  # 15 jours de données
    
    logger.info(f"Récupération des données pour {symbol} sur {timeframe} de {start_date} à {end_date}")
    
    # Récupérer les données
    data = fetcher.fetch_historical_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if data is None or data.empty:
        logger.error("Aucune donnée récupérée")
        return
        
    logger.info(f"Données récupérées: {len(data)} lignes de {data.index.min()} à {data.index.max()}")
    
    # Ajouter les indicateurs
    data_with_indicators = TechnicalIndicators.add_all_indicators(data)
    
    logger.info(f"Données avec indicateurs: {data_with_indicators.shape}")
    
    # Vérifier les indicateurs clés
    indicators_to_check = ['ema_8', 'ema_21', 'macd', 'rsi_14', 'bb_upper']
    
    for ind in indicators_to_check:
        if ind in data_with_indicators.columns:
            non_nan = data_with_indicators[ind].count()
            logger.info(f"Indicateur {ind}: {non_nan} valeurs non-NaN")
        else:
            logger.error(f"Indicateur {ind} manquant")
    
    # Visualiser les prix et quelques indicateurs
    plt.figure(figsize=(12, 8))
    
    # Prix de clôture
    plt.subplot(3, 1, 1)
    plt.plot(data_with_indicators.index, data_with_indicators['close'], label='Prix')
    
    if 'ema_8' in data_with_indicators.columns and 'ema_21' in data_with_indicators.columns:
        plt.plot(data_with_indicators.index, data_with_indicators['ema_8'], label='EMA 8')
        plt.plot(data_with_indicators.index, data_with_indicators['ema_21'], label='EMA 21')
    
    plt.title('Prix et EMA')
    plt.legend()
    
    # RSI
    if 'rsi_14' in data_with_indicators.columns:
        plt.subplot(3, 1, 2)
        plt.plot(data_with_indicators.index, data_with_indicators['rsi_14'])
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('RSI 14')
    
    # MACD
    if 'macd' in data_with_indicators.columns and 'macd_signal' in data_with_indicators.columns:
        plt.subplot(3, 1, 3)
        plt.plot(data_with_indicators.index, data_with_indicators['macd'], label='MACD')
        plt.plot(data_with_indicators.index, data_with_indicators['macd_signal'], label='Signal')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.title('MACD')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_check.png')
    logger.info("Graphique sauvegardé dans data_check.png")
    
    # Compter les croisements potentiels
    if 'ema_8' in data_with_indicators.columns and 'ema_21' in data_with_indicators.columns:
        cross_up = ((data_with_indicators['ema_8'] > data_with_indicators['ema_21']) & 
                   (data_with_indicators['ema_8'].shift(1) <= data_with_indicators['ema_21'].shift(1))).sum()
        cross_down = ((data_with_indicators['ema_8'] < data_with_indicators['ema_21']) & 
                     (data_with_indicators['ema_8'].shift(1) >= data_with_indicators['ema_21'].shift(1))).sum()
                     
        logger.info(f"Croisements EMA 8/21: {cross_up} haussiers, {cross_down} baissiers")

if __name__ == "__main__":
    main()