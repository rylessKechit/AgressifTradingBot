"""
Stratégie de base pour le trading algorithmique
Définit la classe de base que toutes les stratégies doivent étendre
"""
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading
    """
    
    def __init__(self, name, params=None):
        """
        Initialise la stratégie
        
        Args:
            name (str): Nom de la stratégie
            params (dict, optional): Paramètres de la stratégie
        """
        self.name = name
        self.params = params or {}
        logger.info(f"Stratégie {self.name} initialisée avec les paramètres: {self.params}")
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Génère les signaux de trading à partir des données
        
        Args:
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            
        Returns:
            pd.Series: Série contenant les signaux (1=achat, -1=vente, 0=neutre)
        """
        pass
    
    def preprocess_data(self, data):
        """
        Prétraite les données avant de générer les signaux
        Cette méthode peut être surchargée par les sous-classes
        
        Args:
            data (pd.DataFrame): DataFrame avec les données brutes
            
        Returns:
            pd.DataFrame: DataFrame prétraité
        """
        return data
    
    def postprocess_signals(self, signals, data):
        """
        Post-traite les signaux générés
        Cette méthode peut être surchargée par les sous-classes
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux post-traités
        """
        return signals
    
    def run(self, data):
        """
        Exécute la stratégie complète: prétraitement, génération de signaux, post-traitement
        
        Args:
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux finaux
        """
        # Prétraitement des données
        preprocessed_data = self.preprocess_data(data)
        
        # Génération des signaux
        signals = self.generate_signals(preprocessed_data)
        
        # Post-traitement des signaux
        final_signals = self.postprocess_signals(signals, preprocessed_data)
        
        logger.info(f"Stratégie {self.name}: {len(final_signals[final_signals != 0])} signaux générés")
        
        return final_signals
    
    def calculate_dynamic_stop_loss(self, data, position_type, entry_price, atr_multiplier=2.0):
        """
        Calcule un stop loss dynamique basé sur l'ATR
        
        Args:
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            position_type (str): Type de position ('long' ou 'short')
            entry_price (float): Prix d'entrée
            atr_multiplier (float, optional): Multiplicateur pour l'ATR
            
        Returns:
            float: Prix du stop loss
        """
        # Utiliser la dernière valeur de l'ATR
        atr = data['atr_14'].iloc[-1]
        
        if position_type.lower() == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # position_type == 'short'
            stop_loss = entry_price + (atr * atr_multiplier)
            
        return stop_loss
    
    def calculate_take_profit(self, entry_price, stop_loss, position_type, risk_reward_ratio=2.0):
        """
        Calcule un take profit basé sur le ratio risque/récompense
        
        Args:
            entry_price (float): Prix d'entrée
            stop_loss (float): Prix du stop loss
            position_type (str): Type de position ('long' ou 'short')
            risk_reward_ratio (float, optional): Ratio risque/récompense
            
        Returns:
            float: Prix du take profit
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if position_type.lower() == 'long':
            take_profit = entry_price + reward
        else:  # position_type == 'short'
            take_profit = entry_price - reward
            
        return take_profit
    
    def filter_by_trend(self, signals, data, trend_period=50):
        """
        Filtre les signaux en fonction de la tendance
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            trend_period (int, optional): Période pour la tendance
            
        Returns:
            pd.Series: Signaux filtrés
        """
        # Détecter la tendance (utiliser EMA 50 et 200 par exemple)
        trend = np.where(data['ema_50'] > data['ema_200'], 1, -1)
        
        # Filter les signaux contre la tendance
        filtered_signals = signals.copy()
        for i in range(len(signals)):
            # Si signal d'achat mais tendance baissière, annuler le signal
            if signals.iloc[i] == 1 and trend[i] == -1:
                filtered_signals.iloc[i] = 0
            # Si signal de vente mais tendance haussière, annuler le signal
            elif signals.iloc[i] == -1 and trend[i] == 1:
                filtered_signals.iloc[i] = 0
                
        return filtered_signals
    
    def filter_by_volatility(self, signals, data, min_volatility=0.5, max_volatility=2.5):
        """
        Filtre les signaux en fonction de la volatilité
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            min_volatility (float, optional): Volatilité minimale relative
            max_volatility (float, optional): Volatilité maximale relative
            
        Returns:
            pd.Series: Signaux filtrés
        """
        # Calculer la volatilité relative (ATR / moyenne ATR)
        volatility = data['atr_14'] / data['atr_14'].rolling(window=100).mean()
        
        # Filtrer les signaux selon la volatilité
        filtered_signals = signals.copy()
        
        for i in range(len(signals)):
            # Ignorer les signaux pendant les périodes de trop faible ou trop forte volatilité
            if volatility.iloc[i] < min_volatility or volatility.iloc[i] > max_volatility:
                filtered_signals.iloc[i] = 0
                
        return filtered_signals
    
    def filter_by_volume(self, signals, data, volume_threshold=1.5):
        """
        Filtre les signaux en fonction du volume
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            volume_threshold (float, optional): Seuil de volume relatif
            
        Returns:
            pd.Series: Signaux filtrés
        """
        # Calculer le volume relatif (volume / moyenne volume)
        volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Filtrer les signaux selon le volume
        filtered_signals = signals.copy()
        
        for i in range(len(signals)):
            # Conserver uniquement les signaux avec un volume suffisant
            if signals.iloc[i] != 0 and volume_ratio.iloc[i] < volume_threshold:
                filtered_signals.iloc[i] = 0
                
        return filtered_signals
    
    def apply_cooldown(self, signals, cooldown_period=5):
        """
        Applique une période de temporisation après chaque signal
        
        Args:
            signals (pd.Series): Signaux générés
            cooldown_period (int, optional): Nombre de périodes de temporisation
            
        Returns:
            pd.Series: Signaux avec temporisation
        """
        final_signals = signals.copy()
        cooldown_counter = 0
        
        for i in range(len(signals)):
            if cooldown_counter > 0:
                final_signals.iloc[i] = 0
                cooldown_counter -= 1
            elif signals.iloc[i] != 0:
                cooldown_counter = cooldown_period
                
        return final_signals
    
    def get_parameters(self):
        """
        Retourne les paramètres actuels de la stratégie
        
        Returns:
            dict: Paramètres de la stratégie
        """
        return self.params
    
    def set_parameters(self, params):
        """
        Définit de nouveaux paramètres pour la stratégie
        
        Args:
            params (dict): Nouveaux paramètres
        """
        self.params.update(params)
        logger.info(f"Paramètres de la stratégie {self.name} mis à jour: {self.params}")
    
    def evaluate_performance(self, data, signals):
        """
        Évalue la performance de la stratégie (backtest simplifié)
        
        Args:
            data (pd.DataFrame): DataFrame avec les données
            signals (pd.Series): Signaux générés
            
        Returns:
            dict: Métriques de performance
        """
        # Créer un DataFrame pour les résultats
        results = pd.DataFrame(index=data.index)
        results['close'] = data['close']
        results['signal'] = signals
        
        # Calculer les rendements
        results['returns'] = data['close'].pct_change()
        
        # Calculer les rendements de la stratégie
        results['strategy_returns'] = results['returns'] * results['signal'].shift(1)
        
        # Calculer les rendements cumulés
        results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
        results['cumulative_strategy_returns'] = (1 + results['strategy_returns']).cumprod() - 1
        
        # Calculer les métriques de performance
        total_return = results['cumulative_strategy_returns'].iloc[-1]
        
        # Ratio de Sharpe annualisé (en supposant 252 jours de trading par an)
        # et un taux sans risque de 0%
        sharpe_ratio = results['strategy_returns'].mean() / results['strategy_returns'].std() * np.sqrt(252) if results['strategy_returns'].std() > 0 else 0
        
        # Maximum Drawdown
        results['cumulative_peak'] = results['cumulative_strategy_returns'].cummax()
        results['drawdown'] = results['cumulative_peak'] - results['cumulative_strategy_returns']
        max_drawdown = results['drawdown'].max()
        
        # Trades gagnants / perdants
        results['trade'] = results['signal'].diff().fillna(0) != 0
        trade_points = results[results['trade']].index
        trades = []
        
        for i in range(0, len(trade_points) - 1, 2):
            if i + 1 < len(trade_points):
                entry_date = trade_points[i]
                exit_date = trade_points[i + 1]
                entry_price = results.loc[entry_date, 'close']
                exit_price = results.loc[exit_date, 'close']
                trade_type = results.loc[entry_date, 'signal']
                
                if trade_type == 1:  # Long
                    profit = (exit_price / entry_price) - 1
                else:  # Short
                    profit = 1 - (exit_price / entry_price)
                    
                trades.append(profit)
        
        win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_return_per_trade': np.mean(trades) if trades else 0
        }