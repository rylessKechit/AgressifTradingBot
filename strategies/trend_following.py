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
        
        Args:
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            
        Returns:
            pd.Series: Série contenant les signaux (1=achat, -1=vente, 0=neutre)
        """
        # Initialiser les signaux à zéro
        signals = pd.Series(0, index=data.index)
        
        # Récupérer les paramètres
        fast_ema = self.params['fast_ema']
        medium_ema = self.params['medium_ema']
        slow_ema = self.params['slow_ema']
        very_slow_ema = self.params['very_slow_ema']
        
        macd_fast = self.params['macd_fast']
        macd_slow = self.params['macd_slow']
        macd_signal = self.params['macd_signal']
        
        adx_threshold = self.params['adx_threshold']
        
        buy_threshold = self.params['buy_threshold']
        sell_threshold = self.params['sell_threshold']
        
        # --- 1. Signaux basés sur les croisements de moyennes mobiles ---
        
        # Croisements entre EMA rapide et moyenne
        ema_cross_signals = np.zeros(len(data))
        ema_cross_signals[(data[f'ema_{fast_ema}'] > data[f'ema_{medium_ema}']) & 
                        (data[f'ema_{fast_ema}'].shift(1) <= data[f'ema_{medium_ema}'].shift(1))] = 1
        ema_cross_signals[(data[f'ema_{fast_ema}'] < data[f'ema_{medium_ema}']) & 
                        (data[f'ema_{fast_ema}'].shift(1) >= data[f'ema_{medium_ema}'].shift(1))] = -1
        
        # Croisements entre EMA moyenne et lente
        ema_cross_signals[(data[f'ema_{medium_ema}'] > data[f'ema_{slow_ema}']) & 
                        (data[f'ema_{medium_ema}'].shift(1) <= data[f'ema_{slow_ema}'].shift(1))] += 0.5
        ema_cross_signals[(data[f'ema_{medium_ema}'] < data[f'ema_{slow_ema}']) & 
                        (data[f'ema_{medium_ema}'].shift(1) >= data[f'ema_{slow_ema}'].shift(1))] -= 0.5
        
        # --- 2. Signaux basés sur MACD ---
        
        macd_signals = np.zeros(len(data))
        macd_signals[(data['macd'] > data['macd_signal']) & 
                    (data['macd'].shift(1) <= data['macd_signal'].shift(1))] = 1
        macd_signals[(data['macd'] < data['macd_signal']) & 
                    (data['macd'].shift(1) >= data['macd_signal'].shift(1))] = -1
        
        # MACD au-dessus/en-dessous de zéro (confirmation de tendance)
        macd_zero_cross = np.zeros(len(data))
        macd_zero_cross[data['macd'] > 0] = 0.3
        macd_zero_cross[data['macd'] < 0] = -0.3
        
        # --- 3. Signaux basés sur ADX (force de tendance) ---
        
        adx_signals = np.zeros(len(data))
        
        # ADX élevé indique une forte tendance
        strong_trend = data['adx'] > adx_threshold
        
        # Direction de la tendance (+DI/-DI)
        trend_direction = np.zeros(len(data))
        trend_direction[data['adx_plus_di'] > data['adx_minus_di']] = 0.5
        trend_direction[data['adx_plus_di'] < data['adx_minus_di']] = -0.5
        
        # Combiner ADX et direction
        adx_signals = trend_direction * strong_trend
        
        # --- 4. Signaux basés sur la tendance globale (EMA 50/200) ---
        
        trend_signals = np.zeros(len(data))
        
        # Golden Cross / Death Cross
        trend_signals[(data[f'ema_{slow_ema}'] > data[f'ema_{very_slow_ema}']) & 
                    (data[f'ema_{slow_ema}'].shift(1) <= data[f'ema_{very_slow_ema}'].shift(1))] = 1
        trend_signals[(data[f'ema_{slow_ema}'] < data[f'ema_{very_slow_ema}']) & 
                    (data[f'ema_{slow_ema}'].shift(1) >= data[f'ema_{very_slow_ema}'].shift(1))] = -1
        
        # Tendance actuelle
        current_trend = np.zeros(len(data))
        current_trend[data[f'ema_{slow_ema}'] > data[f'ema_{very_slow_ema}']] = 0.2
        current_trend[data[f'ema_{slow_ema}'] < data[f'ema_{very_slow_ema}']] = -0.2
        
        # --- 5. Combinaison des signaux ---
        
        # Pondération des différents signaux
        ema_weight = 0.4
        macd_weight = 0.3
        adx_weight = 0.2
        trend_weight = 0.1
        
        # Signal combiné
        combined_signals = (
            ema_cross_signals * ema_weight + 
            (macd_signals + macd_zero_cross) * macd_weight + 
            adx_signals * adx_weight + 
            (trend_signals + current_trend) * trend_weight
        )
        
        # --- 6. Génération des signaux finaux basés sur les seuils ---
        
        signals[combined_signals > buy_threshold] = 1  # Signal d'achat
        signals[combined_signals < sell_threshold] = -1  # Signal de vente
        
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