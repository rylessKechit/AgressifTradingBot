"""
Stratégie de retour à la moyenne (Mean Reversion)
Implémente une stratégie basée sur le retour à la moyenne en utilisant des
indicateurs comme RSI, bandes de Bollinger, stochastique, etc.
"""
import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from config.strategy_params import MEAN_REVERSION_PARAMS

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie qui exploite le phénomène de retour à la moyenne des prix
    après des mouvements extrêmes
    """
    
    def __init__(self, params=None):
        """
        Initialise la stratégie de retour à la moyenne
        
        Args:
            params (dict, optional): Paramètres personnalisés pour la stratégie
        """
        # Charger les paramètres par défaut
        default_params = MEAN_REVERSION_PARAMS.copy()
        
        # Fusionner avec les paramètres personnalisés s'ils existent
        if params:
            default_params.update(params)
        
        super().__init__("Mean Reversion", default_params)
    
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
            'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 
            'stoch_k', 'stoch_d', 'cci_14', 'atr_14'
        ]
        
        # Si des indicateurs sont manquants, renvoyer un avertissement
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        if missing_indicators:
            logger.warning(f"Indicateurs manquants pour la stratégie de retour à la moyenne: {missing_indicators}")
            logger.warning("Assurez-vous que tous les indicateurs sont calculés avant d'utiliser cette stratégie")
        
        return data
    
    def generate_signals(self, data):
        """
        Génère les signaux de trading basés sur le retour à la moyenne
        
        Args:
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            
        Returns:
            pd.Series: Série contenant les signaux (1=achat, -1=vente, 0=neutre)
        """
        # Initialiser les signaux à zéro
        signals = pd.Series(0, index=data.index)
        
        # Récupérer les paramètres
        rsi_period = self.params['rsi_period']
        rsi_overbought = self.params['rsi_overbought']
        rsi_oversold = self.params['rsi_oversold']
        
        bb_period = self.params['bb_period']
        bb_std = self.params['bb_std']
        
        stoch_overbought = self.params['stoch_overbought']
        stoch_oversold = self.params['stoch_oversold']
        
        cci_overbought = self.params['cci_overbought']
        cci_oversold = self.params['cci_oversold']
        
        buy_threshold = self.params['buy_threshold']
        sell_threshold = self.params['sell_threshold']
        
        # --- 1. Signaux basés sur le RSI ---
        
        rsi_signals = np.zeros(len(data))
        
        # Conditions de surachat/survente
        rsi_signals[data['rsi_14'] < rsi_oversold] = 1
        rsi_signals[data['rsi_14'] > rsi_overbought] = -1
        
        # Divergences (optionnel)
        # Divergence haussière: prix baisse mais RSI monte
        rsi_signals[(data['close'] < data['close'].shift(1)) & 
                   (data['rsi_14'] > data['rsi_14'].shift(1)) & 
                   (data['rsi_14'] < 50)] += 0.5
        
        # Divergence baissière: prix monte mais RSI baisse
        rsi_signals[(data['close'] > data['close'].shift(1)) & 
                   (data['rsi_14'] < data['rsi_14'].shift(1)) & 
                   (data['rsi_14'] > 50)] -= 0.5
        
        # --- 2. Signaux basés sur les Bandes de Bollinger ---
        
        bb_signals = np.zeros(len(data))
        
        # Dépassement des bandes
        bb_signals[data['close'] < data['bb_lower']] = 1
        bb_signals[data['close'] > data['bb_upper']] = -1
        
        # Retour à l'intérieur des bandes (confirmation)
        bb_signals[(data['close'] > data['bb_lower']) & 
                  (data['close'].shift(1) <= data['bb_lower'].shift(1))] += 0.5
        bb_signals[(data['close'] < data['bb_upper']) & 
                  (data['close'].shift(1) >= data['bb_upper'].shift(1))] -= 0.5
        
        # Compression des bandes (squeeze) suivi d'une expansion
        bb_width = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        bb_width_ma = bb_width.rolling(window=20).mean()
        
        squeeze = (bb_width < bb_width_ma * 0.8).astype(int)
        squeeze_release = ((squeeze.shift(1) == 1) & (squeeze == 0)).astype(int)
        
        # Ne pas générer de signal spécifique pour le squeeze, mais l'utiliser comme filtre
        
        # --- 3. Signaux basés sur le Stochastique ---
        
        stoch_signals = np.zeros(len(data))
        
        # Survente/surachat
        stoch_signals[(data['stoch_k'] < stoch_oversold) & (data['stoch_d'] < stoch_oversold)] = 1
        stoch_signals[(data['stoch_k'] > stoch_overbought) & (data['stoch_d'] > stoch_overbought)] = -1
        
        # Croisement
        stoch_signals[(data['stoch_k'] > data['stoch_d']) & 
                     (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) & 
                     (data['stoch_k'] < 50)] += 0.5
        stoch_signals[(data['stoch_k'] < data['stoch_d']) & 
                     (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) & 
                     (data['stoch_k'] > 50)] -= 0.5
        
        # --- 4. Signaux basés sur le CCI ---
        
        cci_signals = np.zeros(len(data))
        
        # Survente/surachat
        cci_signals[data['cci_14'] < cci_oversold] = 1
        cci_signals[data['cci_14'] > cci_overbought] = -1
        
        # Retour vers zéro depuis des extrêmes
        cci_signals[(data['cci_14'] > -100) & (data['cci_14'].shift(1) <= -100)] += 0.3
        cci_signals[(data['cci_14'] < 100) & (data['cci_14'].shift(1) >= 100)] -= 0.3
        
        # --- 5. Combinaison des signaux ---
        
        # Pondération des différents signaux
        rsi_weight = 0.35
        bb_weight = 0.35
        stoch_weight = 0.2
        cci_weight = 0.1
        
        # Signal combiné
        combined_signals = (
            rsi_signals * rsi_weight + 
            bb_signals * bb_weight + 
            stoch_signals * stoch_weight + 
            cci_signals * cci_weight
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
        # Filtrer par volatilité si demandé
        if self.params.get('use_volatility_filter', True):
            signals = self.filter_by_volatility(signals, data)
        
        # Appliquer une temporisation entre les signaux
        signals = self.apply_cooldown(signals, cooldown_period=self.params.get('cooldown_period', 5))
        
        # Filtrer par volume si demandé
        if self.params.get('use_volume_filter', True):
            signals = self.filter_by_volume(signals, data, volume_threshold=self.params.get('volume_threshold', 1.2))
        
        return signals
    
    def filter_by_volatility(self, signals, data):
        """
        Filtre les signaux en fonction de la volatilité
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux filtrés
        """
        filtered_signals = signals.copy()
        
        # Calculer la volatilité relative (ATR / moyenne ATR)
        atr_period = self.params.get('atr_period', 14)
        atr_column = f'atr_{atr_period}'
        
        if atr_column not in data.columns:
            logger.warning(f"Colonne {atr_column} manquante pour le filtrage par volatilité")
            return signals
        
        volatility = data[atr_column] / data[atr_column].rolling(window=100).mean()
        
        # Seuils de volatilité
        min_volatility = 0.5
        max_volatility = 2.0
        
        for i in range(len(signals)):
            # Ignorer les signaux pendant les périodes de trop forte volatilité (pour retour à la moyenne)
            if volatility.iloc[i] > max_volatility and signals.iloc[i] != 0:
                filtered_signals.iloc[i] = 0
            
            # Lorsque la volatilité est très faible, nous avons généralement moins de retours à la moyenne
            if volatility.iloc[i] < min_volatility and signals.iloc[i] != 0:
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
    strategy = MeanReversionStrategy()
    
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
    plt.figure(figsize=(14, 10))
    
    # Subplot pour le prix et les bandes de Bollinger
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Prix', alpha=0.7)
    plt.plot(data.index, data['bb_upper'], 'r--', label='BB Upper')
    plt.plot(data.index, data['bb_middle'], 'g--', label='BB Middle')
    plt.plot(data.index, data['bb_lower'], 'r--', label='BB Lower')
    
    # Tracer les signaux
    buy_signals = data[signals == 1].index
    sell_signals = data[signals == -1].index
    
    plt.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Achat')
    plt.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Vente')
    
    plt.title('Stratégie de retour à la moyenne - Prix et Bandes de Bollinger')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour le RSI
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['rsi_14'], label='RSI 14')
    plt.axhline(y=strategy.params['rsi_oversold'], color='g', linestyle='--', label=f'Survente ({strategy.params["rsi_oversold"]})')
    plt.axhline(y=strategy.params['rsi_overbought'], color='r', linestyle='--', label=f'Surachat ({strategy.params["rsi_overbought"]})')
    plt.axhline(y=50, color='k', linestyle='-', alpha=0.2)
    
    plt.title('RSI')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour le Stochastique
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['stoch_k'], label='Stoch %K')
    plt.plot(data.index, data['stoch_d'], label='Stoch %D')
    plt.axhline(y=strategy.params['stoch_oversold'], color='g', linestyle='--', label=f'Survente ({strategy.params["stoch_oversold"]})')
    plt.axhline(y=strategy.params['stoch_overbought'], color='r', linestyle='--', label=f'Surachat ({strategy.params["stoch_overbought"]})')
    
    plt.title('Stochastique')
    plt.xlabel('Date')
    plt.ylabel('Stochastique')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()