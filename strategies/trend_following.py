"""
Stratégie de suivi de tendance
Implémente une stratégie de suivi de tendance basée sur les moyennes mobiles,
MACD et autres indicateurs de tendance
"""
import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from config.strategy_params import TREND_FOLLOWING_PARAMS

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    Stratégie de trading qui suit la tendance en utilisant des moyennes mobiles,
    MACD et autres indicateurs de tendance
    """
    
    def __init__(self, params=None):
        """
        Initialise la stratégie de suivi de tendance
        
        Args:
            params (dict, optional): Paramètres personnalisés pour la stratégie
        """
        # Charger les paramètres par défaut
        default_params = TREND_FOLLOWING_PARAMS.copy()
        
        # Fusionner avec les paramètres personnalisés s'ils existent
        if params:
            default_params.update(params)
        
        super().__init__("Trend Following", default_params)
    
    def preprocess_data(self, data):
        """
        Prétraite les données pour la stratégie
        
        Args:
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.DataFrame: DataFrame prétraité
        """
        # Vérifier si les indicateurs nécessaires sont déjà calculés
        required_indicators = [
            'ema_8', 'ema_21', 'ema_50', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'adx'
        ]
        
        # Si des indicateurs sont manquants, renvoyer un avertissement
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        if missing_indicators:
            logger.warning(f"Indicateurs manquants pour la stratégie de suivi de tendance: {missing_indicators}")
            logger.warning("Assurez-vous que tous les indicateurs sont calculés avant d'utiliser cette stratégie")
        
        return data
    
    def generate_signals(self, data):
        """
        Génère les signaux de trading basés sur la tendance
        """
        # Initialiser les signaux à zéro
        signals = pd.Series(0, index=data.index)
        
        # Version simplifiée et plus sensible
        for i in range(1, len(data)):
            # Signal d'achat: croisement EMA 8 au-dessus de EMA 21 + MACD positif
            if (data['ema_8'].iloc[i] > data['ema_21'].iloc[i] and 
                data['ema_8'].iloc[i-1] <= data['ema_21'].iloc[i-1] and
                data['macd'].iloc[i] > 0):
                signals.iloc[i] = 1
                
            # Signal de vente: croisement EMA 8 en-dessous de EMA 21 + MACD négatif
            elif (data['ema_8'].iloc[i] < data['ema_21'].iloc[i] and 
                data['ema_8'].iloc[i-1] >= data['ema_21'].iloc[i-1] and
                data['macd'].iloc[i] < 0):
                signals.iloc[i] = -1
                
            # Signaux supplémentaires basés uniquement sur MACD
            elif (data['macd'].iloc[i] > data['macd_signal'].iloc[i] and 
                data['macd'].iloc[i-1] <= data['macd_signal'].iloc[i-1]):
                signals.iloc[i] = 1
                
            elif (data['macd'].iloc[i] < data['macd_signal'].iloc[i] and 
                data['macd'].iloc[i-1] >= data['macd_signal'].iloc[i-1]):
                signals.iloc[i] = -1
        
        return signals

    def postprocess_signals(self, signals, data):
        """
        Post-traite les signaux générés pour éliminer les faux signaux
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux post-traités
        """
        # Filtrer les signaux contre la tendance principale (optionnel selon la stratégie)
        if self.params.get('filter_against_main_trend', True):
            signals = self.filter_by_trend(signals, data)
        
        # Appliquer une temporisation entre les signaux
        signals = self.apply_cooldown(signals, cooldown_period=self.params.get('cooldown_period', 3))
        
        # Filtrer par volume si demandé
        if self.params.get('use_volume_filter', True):
            signals = self.filter_by_volume(signals, data, volume_threshold=self.params.get('volume_threshold', 1.2))
        
        return signals
    
    def filter_by_trend(self, signals, data):
        """
        Filtre les signaux pour s'assurer qu'ils suivent la tendance principale
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux filtrés
        """
        filtered_signals = signals.copy()
        
        # Déterminer la tendance principale (EMA 50 vs EMA 200)
        main_trend = np.where(data[f'ema_{self.params["slow_ema"]}'] > data[f'ema_{self.params["very_slow_ema"]}'], 1, -1)
        
        # Ne conserver que les signaux qui suivent la tendance principale
        for i in range(len(signals)):
            # Si signal d'achat mais tendance baissière
            if signals.iloc[i] == 1 and main_trend[i] == -1:
                # Vérifier si l'ADX est faible (marché sans tendance forte)
                if data['adx'].iloc[i] < self.params['adx_threshold']:
                    # Conserver le signal car le marché n'a pas de tendance forte
                    pass
                else:
                    # Annuler le signal car il va contre une tendance baissière forte
                    filtered_signals.iloc[i] = 0
            
            # Si signal de vente mais tendance haussière
            elif signals.iloc[i] == -1 and main_trend[i] == 1:
                # Vérifier si l'ADX est faible (marché sans tendance forte)
                if data['adx'].iloc[i] < self.params['adx_threshold']:
                    # Conserver le signal car le marché n'a pas de tendance forte
                    pass
                else:
                    # Annuler le signal car il va contre une tendance haussière forte
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from data.fetcher import DataFetcher
    from data.indicators import TechnicalIndicators
    from datetime import datetime, timedelta
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Récupérer des données historiques
    fetcher = DataFetcher()
    
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()
    
    data = fetcher.fetch_historical_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date
    )
    
    # Ajouter les indicateurs techniques
    data = TechnicalIndicators.add_all_indicators(data)
    
    # Initialiser la stratégie
    strategy = TrendFollowingStrategy()
    
    # Générer les signaux
    signals = strategy.run(data)
    
    # Évaluer la performance
    performance = strategy.evaluate_performance(data, signals)
    
    # Afficher les résultats
    print(f"Rendement total: {performance['total_return']:.2%}")
    print(f"Ratio de Sharpe: {performance['sharpe_ratio']:.2f}")
    print(f"Drawdown maximum: {performance['max_drawdown']:.2%}")
    print(f"Taux de réussite: {performance['win_rate']:.2%}")
    print(f"Nombre de trades: {performance['num_trades']}")
    
    # Tracer un graphique
    plt.figure(figsize=(14, 7))
    
    # Tracer le prix
    plt.plot(data.index, data['close'], label='Prix', alpha=0.5)
    
    # Tracer les moyennes mobiles
    plt.plot(data.index, data[f'ema_{strategy.params["fast_ema"]}'], label=f'EMA {strategy.params["fast_ema"]}')
    plt.plot(data.index, data[f'ema_{strategy.params["medium_ema"]}'], label=f'EMA {strategy.params["medium_ema"]}')
    plt.plot(data.index, data[f'ema_{strategy.params["slow_ema"]}'], label=f'EMA {strategy.params["slow_ema"]}')
    
    # Tracer les signaux
    buy_signals = data[signals == 1].index
    sell_signals = data[signals == -1].index
    
    plt.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Achat')
    plt.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Vente')
    
    plt.title('Stratégie de suivi de tendance')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    plt.show()