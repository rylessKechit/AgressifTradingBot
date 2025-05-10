"""
Module de calcul de taille de position
Gère le calcul de la taille des positions en fonction du risque
"""
import logging
import numpy as np
from config.settings import CAPITAL, TRADING

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Classe pour calculer la taille des positions en fonction du risque
    """
    
    def __init__(self, initial_capital=None, max_risk_per_trade=None, max_position_size=None):
        """
        Initialise le PositionSizer
        
        Args:
            initial_capital (float, optional): Capital initial
            max_risk_per_trade (float, optional): Risque maximum par trade (en %)
            max_position_size (float, optional): Taille maximum de position (en % du capital)
        """
        self.initial_capital = initial_capital or CAPITAL.get('initial_capital', 10000)
        self.max_risk_per_trade = max_risk_per_trade or CAPITAL.get('risk_per_trade', 0.01)
        self.max_position_size = max_position_size or CAPITAL.get('max_position_size', 0.2)
        
        logger.info(f"PositionSizer initialisé avec capital={self.initial_capital}, " +
                   f"max_risk={self.max_risk_per_trade*100}%, max_position={self.max_position_size*100}%")
    
    def calculate_position_size(self, capital, price, stop_loss_percent, risk_percent=None):
        """
        Calcule la taille de position optimale selon la gestion des risques
        
        Args:
            capital (float): Capital actuel
            price (float): Prix actuel
            stop_loss_percent (float): Pourcentage du stop loss
            risk_percent (float, optional): Pourcentage du capital à risquer
            
        Returns:
            float: Taille de la position en unités
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
            
        # Montant à risquer
        risk_amount = capital * risk_percent
        
        # Calcul de la taille de position
        if stop_loss_percent == 0:
            logger.warning("Stop loss à 0%, impossible de calculer la taille de position")
            return 0
            
        position_value = risk_amount / stop_loss_percent
        
        # Limiter la taille de position au maximum autorisé
        max_position_value = capital * self.max_position_size
        if position_value > max_position_value:
            logger.warning(f"Taille de position calculée ({position_value}) supérieure au maximum autorisé ({max_position_value}), limitée")
            position_value = max_position_value
        
        # Convertir en unités
        position_size = position_value / price
        
        # Arrondir à la précision appropriée
        position_size = round(position_size, 6)
        
        logger.info(f"Taille de position calculée: {position_size} unités, valeur: {position_value:.2f}, " +
                   f"risque: {risk_amount:.2f} ({risk_percent*100}%), stop: {stop_loss_percent*100}%")
        
        return position_size
    
    def calculate_position_size_atr(self, capital, price, atr, atr_multiplier=2.0, risk_percent=None):
        """
        Calcule la taille de position optimale selon l'ATR
        
        Args:
            capital (float): Capital actuel
            price (float): Prix actuel
            atr (float): Average True Range
            atr_multiplier (float, optional): Multiplicateur de l'ATR
            risk_percent (float, optional): Pourcentage du capital à risquer
            
        Returns:
            float: Taille de la position en unités
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
            
        # Montant à risquer
        risk_amount = capital * risk_percent
        
        # Calcul du stop loss basé sur l'ATR
        stop_loss_amount = atr * atr_multiplier
        
        # Calcul de la taille de position
        if stop_loss_amount == 0:
            logger.warning("ATR à 0, impossible de calculer la taille de position")
            return 0
            
        position_size = risk_amount / stop_loss_amount
        
        # Convertir en unités
        position_units = position_size / price
        
        # Limiter la taille de position au maximum autorisé
        max_position_value = capital * self.max_position_size
        max_position_units = max_position_value / price
        
        if position_units > max_position_units:
            logger.warning(f"Taille de position calculée ({position_units}) supérieure au maximum autorisé ({max_position_units}), limitée")
            position_units = max_position_units
        
        # Arrondir à la précision appropriée
        position_units = round(position_units, 6)
        
        logger.info(f"Taille de position calculée avec ATR: {position_units} unités, " +
                   f"risque: {risk_amount:.2f} ({risk_percent*100}%), stop: {stop_loss_amount:.2f} ({atr_multiplier}*ATR)")
        
        return position_units
    
    def adjust_position_size_by_volatility(self, position_size, current_volatility, avg_volatility):
        """
        Ajuste la taille de position en fonction de la volatilité
        
        Args:
            position_size (float): Taille de position de base
            current_volatility (float): Volatilité actuelle
            avg_volatility (float): Volatilité moyenne
            
        Returns:
            float: Taille de position ajustée
        """
        if avg_volatility == 0:
            logger.warning("Volatilité moyenne à 0, impossible d'ajuster la taille de position")
            return position_size
            
        # Calculer le ratio de volatilité
        volatility_ratio = current_volatility / avg_volatility
        
        # Ajuster la taille de position inversement à la volatilité
        # Plus la volatilité est élevée, plus la taille de position est réduite
        adjustment_factor = 1 / volatility_ratio
        
        # Limiter l'ajustement
        min_adjustment = 0.5  # Réduction maximale de 50%
        max_adjustment = 2.0  # Augmentation maximale de 200%
        
        adjustment_factor = max(min_adjustment, min(adjustment_factor, max_adjustment))
        
        # Appliquer l'ajustement
        adjusted_position_size = position_size * adjustment_factor
        
        logger.info(f"Taille de position ajustée par volatilité: {position_size} -> {adjusted_position_size}, " +
                   f"volatilité: {current_volatility:.4f}/{avg_volatility:.4f} ({volatility_ratio:.2f}), " +
                   f"facteur: {adjustment_factor:.2f}")
        
        return adjusted_position_size
    
    def kelly_criterion(self, win_rate, reward_risk_ratio):
        """
        Calcule la taille de position optimale selon le critère de Kelly
        
        Args:
            win_rate (float): Taux de réussite (0-1)
            reward_risk_ratio (float): Ratio récompense/risque
            
        Returns:
            float: Fraction de Kelly (% du capital à risquer)
        """
        # Formule de Kelly: f* = (p*b - q) / b
        # où p = probabilité de gain, q = probabilité de perte (1-p), b = ratio récompense/risque
        
        q = 1 - win_rate
        kelly_fraction = (win_rate * reward_risk_ratio - q) / reward_risk_ratio
        
        # Limiter la fraction de Kelly
        kelly_fraction = max(0, min(kelly_fraction, self.max_risk_per_trade))
        
        logger.info(f"Fraction de Kelly calculée: {kelly_fraction:.4f}, " +
                   f"win_rate: {win_rate:.2f}, reward/risk: {reward_risk_ratio:.2f}")
        
        return kelly_fraction
    
    def optimal_f(self, trades, protection_factor=0.5):
        """
        Calcule la taille de position optimale selon la méthode de l'optimal f de Ralph Vince
        
        Args:
            trades (list): Liste des trades (pertes/gains relatifs)
            protection_factor (float, optional): Facteur de protection (0-1)
            
        Returns:
            float: Fraction optimale (% du capital à risquer)
        """
        if not trades:
            logger.warning("Pas de trades pour calculer l'optimal f")
            return self.max_risk_per_trade
            
        # Convertir les trades en liste de nombres
        trades = np.array(trades)
        
        # Trouver la perte maximale (en valeur absolue)
        worst_loss = abs(min(trades))
        
        if worst_loss == 0:
            logger.warning("Pas de pertes dans les trades, impossible de calculer l'optimal f")
            return self.max_risk_per_trade
            
        # Calculer l'optimal f
        optimal_f = 0
        best_result = 0
        
        # Tester différentes valeurs de f
        for f in np.arange(0.01, 1.01, 0.01):
            # Calculer le facteur de croissance du capital
            TWR = 1.0
            for t in trades:
                # Éviter la ruine
                if f >= 1 / worst_loss:
                    TWR = 0
                    break
                    
                # Calculer le facteur de croissance pour ce trade
                TWR *= (1 + f * t)
                
            # Mettre à jour l'optimal f si le résultat est meilleur
            if TWR > best_result:
                best_result = TWR
                optimal_f = f
        
        # Appliquer le facteur de protection
        safe_f = optimal_f * protection_factor
        
        # Limiter la fraction optimale
        safe_f = min(safe_f, self.max_risk_per_trade)
        
        logger.info(f"Optimal f calculé: {optimal_f:.4f}, sécurisé: {safe_f:.4f} (facteur: {protection_factor})")
        
        return safe_f
    
    def dynamic_position_sizing(self, capital, base_risk, equity_curve):
        """
        Ajuste dynamiquement le risque en fonction de la performance
        
        Args:
            capital (float): Capital actuel
            base_risk (float): Risque de base (%)
            equity_curve (list): Courbe d'équité
            
        Returns:
            float: Risque ajusté (%)
        """
        if not equity_curve or len(equity_curve) < 10:
            logger.info("Pas assez de données pour ajuster dynamiquement le risque")
            return base_risk
            
        # Convertir la courbe d'équité en liste
        if isinstance(equity_curve, list):
            equity = equity_curve
        else:
            equity = list(equity_curve)
            
        # Calculer les rendements
        returns = []
        for i in range(1, len(equity)):
            returns.append((equity[i] - equity[i-1]) / equity[i-1])
            
        # Calculer la moyenne et l'écart-type des rendements
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculer le ratio de Sharpe (simplifié)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Ajuster le risque en fonction du ratio de Sharpe
        if sharpe_ratio > 1.0:  # Bon ratio
            risk_factor = 1.2  # Augmenter le risque
        elif sharpe_ratio > 0.5:  # Ratio acceptable
            risk_factor = 1.0  # Maintenir le risque
        else:  # Mauvais ratio
            risk_factor = 0.8  # Réduire le risque
            
        # Ajuster en fonction de la tendance récente
        recent_returns = returns[-10:]  # 10 derniers rendements
        recent_mean = np.mean(recent_returns)
        
        if recent_mean > mean_return:  # Tendance positive
            risk_factor *= 1.1
        elif recent_mean < 0:  # Tendance négative
            risk_factor *= 0.9
            
        # Calculer le risque ajusté
        adjusted_risk = base_risk * risk_factor
        
        # Limiter le risque
        adjusted_risk = max(base_risk * 0.5, min(adjusted_risk, base_risk * 1.5))
        
        logger.info(f"Risque ajusté dynamiquement: {base_risk:.4f} -> {adjusted_risk:.4f}, " +
                   f"Sharpe: {sharpe_ratio:.2f}, tendance récente: {recent_mean:.4f}")
        
        return adjusted_risk


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialiser le calculateur de taille de position
    position_sizer = PositionSizer(initial_capital=10000, max_risk_per_trade=0.01, max_position_size=0.2)
    
    # Calculer la taille de position avec un stop loss fixe
    position_size = position_sizer.calculate_position_size(10000, 50000, 0.03)
    print(f"Taille de position avec stop loss fixe: {position_size} BTC (valeur: {position_size * 50000} USD)")
    
    # Calculer la taille de position avec ATR
    atr = 1500  # ATR en USD pour BTC
    position_size_atr = position_sizer.calculate_position_size_atr(10000, 50000, atr, 2.0)
    print(f"Taille de position avec ATR: {position_size_atr} BTC (valeur: {position_size_atr * 50000} USD)")
    
    # Ajuster la taille de position en fonction de la volatilité
    current_vol = 0.03  # 3% de volatilité quotidienne
    avg_vol = 0.02  # 2% de volatilité moyenne
    adjusted_size = position_sizer.adjust_position_size_by_volatility(position_size, current_vol, avg_vol)
    print(f"Taille de position ajustée par volatilité: {adjusted_size} BTC (valeur: {adjusted_size * 50000} USD)")
    
    # Calculer la fraction de Kelly
    win_rate = 0.55
    reward_risk = 1.5
    kelly = position_sizer.kelly_criterion(win_rate, reward_risk)
    print(f"Fraction de Kelly: {kelly:.4f} ({kelly*100}% du capital)")
    
    # Calculer l'optimal f
    trades = [0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01, 0.03, -0.02]
    opt_f = position_sizer.optimal_f(trades, 0.5)
    print(f"Optimal f: {opt_f:.4f} ({opt_f*100}% du capital)")
    
    # Ajuster dynamiquement la taille de position
    equity_curve = [10000, 10200, 10350, 10300, 10450, 10550, 10500, 10650, 10700, 10800, 10750]
    dyn_risk = position_sizer.dynamic_position_sizing(10800, 0.01, equity_curve)
    print(f"Risque ajusté dynamiquement: {dyn_risk:.4f} ({dyn_risk*100}% du capital)")