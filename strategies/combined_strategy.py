"""
Stratégie combinée 
Implémente une stratégie qui combine plusieurs approches (suivi de tendance,
retour à la moyenne, arbitrage) pour améliorer la robustesse
"""
import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.arbitrage import ArbitrageStrategy  # À implémenter
from config.strategy_params import COMBINED_STRATEGY_PARAMS

logger = logging.getLogger(__name__)

class CombinedStrategy(BaseStrategy):
    """
    Stratégie qui combine plusieurs approches pour améliorer la robustesse et
    s'adapter à différentes conditions de marché
    """
    
    def __init__(self, params=None):
        """
        Initialise la stratégie combinée
        
        Args:
            params (dict, optional): Paramètres personnalisés pour la stratégie
        """
        # Charger les paramètres par défaut
        default_params = COMBINED_STRATEGY_PARAMS.copy()
        
        # Fusionner avec les paramètres personnalisés s'ils existent
        if params:
            default_params.update(params)
        
        super().__init__("Combined Strategy", default_params)
        
        # Initialiser les sous-stratégies
        self.trend_strategy = TrendFollowingStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        
        # Initialiser la stratégie d'arbitrage si elle est implémentée
        try:
            self.arbitrage_strategy = ArbitrageStrategy()
            self.has_arbitrage = True
        except:
            logger.warning("Stratégie d'arbitrage non implémentée, sera ignorée dans la combinaison")
            self.has_arbitrage = False
    
    def preprocess_data(self, data):
        """
        Prétraite les données pour la stratégie
        
        Args:
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.DataFrame: DataFrame prétraité
        """
        # Les stratégies individuelles se chargeront de vérifier leurs propres indicateurs
        return data
    
    def generate_signals(self, data):
        """
        Génère les signaux de trading en combinant plusieurs stratégies
        """
        # Initialiser les signaux à zéro
        signals = pd.Series(0, index=data.index)
        
        # Obtenir les signaux des sous-stratégies
        trend_signals = self.trend_strategy.run(data)
        mean_reversion_signals = self.mean_reversion_strategy.run(data)
        
        if self.has_arbitrage:
            arbitrage_signals = self.arbitrage_strategy.run(data)
        else:
            arbitrage_signals = pd.Series(0, index=data.index)
        
        # Récupérer les poids des stratégies depuis les configurations
        trend_weight = self.trend_strategy.params.get('weight', 0.4)
        mean_reversion_weight = self.mean_reversion_strategy.params.get('weight', 0.4)
        arbitrage_weight = 0.2 if self.has_arbitrage else 0.0
        
        # Combinaison simple des signaux avec poids fixes
        combined_signals = (
            trend_signals * trend_weight +
            mean_reversion_signals * mean_reversion_weight +
            arbitrage_signals * arbitrage_weight
        )
        
        # Abaisser les seuils pour générer plus de signaux
        signals[combined_signals > 0.3] = 1  # Signal d'achat (seuil abaissé)
        signals[combined_signals < -0.3] = -1  # Signal de vente (seuil abaissé)
        
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
        # Appliquer une temporisation entre les signaux
        signals = self.apply_cooldown(signals, cooldown_period=self.params.get('cooldown_period', 3))
        
        # Filtrer par volume si demandé
        if self.params.get('use_volume_filter', True):
            signals = self.filter_by_volume(signals, data, volume_threshold=self.params.get('volume_threshold', 1.5))
        
        # Vérification multi-timeframes si demandé
        if self.params.get('use_multi_timeframe', True) and 'tf_alignment' in data.columns:
            signals = self.filter_by_timeframe_alignment(signals, data)
        
        return signals
    
    def filter_by_timeframe_alignment(self, signals, data):
        """
        Filtre les signaux en fonction de l'alignement des timeframes
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux filtrés
        """
        filtered_signals = signals.copy()
        
        # Si les données contiennent des informations d'alignement de timeframe
        if 'tf_alignment' in data.columns:
            alignment_required = self.params.get('timeframe_alignment_required', False)
            
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    # Si l'alignement est requis mais non présent, annuler le signal
                    if alignment_required and data['tf_alignment'].iloc[i] == 0:
                        filtered_signals.iloc[i] = 0
                    # Sinon, utiliser l'alignement pour renforcer ou réduire la confiance dans le signal
                    else:
                        signal_strength = data['tf_alignment'].iloc[i] * signals.iloc[i]
                        if signal_strength <= 0:  # Si l'alignement est contradictoire
                            filtered_signals.iloc[i] = 0
        
        return filtered_signals


# Exemple d'utilisation
if __name__ == "__main__":
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
    
    # Initialiser la stratégie combinée
    strategy = CombinedStrategy()
    
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
    
    # Tracer le prix
    plt.plot(data.index, data['close'], label='Prix', alpha=0.5)
    
    # Tracer les moyennes mobiles
    plt.plot(data.index, data['ema_50'], label='EMA 50')
    plt.plot(data.index, data['ema_200'], label='EMA 200')
    
    # Tracer les signaux
    buy_signals = data[signals == 1].index
    sell_signals = data[signals == -1].index
    
    plt.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Achat')
    plt.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Vente')
    
    plt.title('Stratégie combinée')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    plt.show()