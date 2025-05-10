"""
Module de calcul des niveaux de stop loss
Fournit différentes stratégies pour déterminer les niveaux de stop loss
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class StopLossCalculator:
    """
    Classe pour calculer les niveaux de stop loss selon différentes stratégies
    """
    
    def __init__(self, default_percent=0.03, atr_multiplier=2.0):
        """
        Initialise le calculateur de stop loss
        
        Args:
            default_percent (float, optional): Pourcentage par défaut pour le stop loss
            atr_multiplier (float, optional): Multiplicateur pour la méthode ATR
        """
        self.default_percent = default_percent
        self.atr_multiplier = atr_multiplier
        
        logger.info(f"StopLossCalculator initialisé avec: default_percent={default_percent}, atr_multiplier={atr_multiplier}")
    
    def calculate_fixed_percent(self, entry_price, side):
        """
        Calcule un stop loss basé sur un pourcentage fixe
        
        Args:
            entry_price (float): Prix d'entrée
            side (str): Direction de la position ('long' ou 'short')
            
        Returns:
            float: Niveau de stop loss
        """
        if side.lower() == 'long':
            stop_level = entry_price * (1 - self.default_percent)
        else:  # short
            stop_level = entry_price * (1 + self.default_percent)
            
        logger.debug(f"Stop loss fixe calculé: {stop_level} (entrée: {entry_price}, côté: {side})")
        return stop_level
    
    def calculate_atr_stop(self, entry_price, atr_value, side):
        """
        Calcule un stop loss basé sur l'ATR (Average True Range)
        
        Args:
            entry_price (float): Prix d'entrée
            atr_value (float): Valeur ATR
            side (str): Direction de la position ('long' ou 'short')
            
        Returns:
            float: Niveau de stop loss
        """
        stop_distance = atr_value * self.atr_multiplier
        
        if side.lower() == 'long':
            stop_level = entry_price - stop_distance
        else:  # short
            stop_level = entry_price + stop_distance
            
        logger.debug(f"Stop loss ATR calculé: {stop_level} (entrée: {entry_price}, ATR: {atr_value}, côté: {side})")
        return stop_level
    
    def calculate_support_resistance_stop(self, entry_price, support_levels, resistance_levels, side):
        """
        Calcule un stop loss basé sur les niveaux de support et résistance
        
        Args:
            entry_price (float): Prix d'entrée
            support_levels (list): Niveaux de support
            resistance_levels (list): Niveaux de résistance
            side (str): Direction de la position ('long' ou 'short')
            
        Returns:
            float: Niveau de stop loss
        """
        if side.lower() == 'long':
            # Pour une position longue, trouver le support le plus proche en dessous du prix d'entrée
            valid_supports = [s for s in support_levels if s < entry_price]
            
            if valid_supports:
                # Prendre le support le plus proche
                stop_level = max(valid_supports)
            else:
                # Si pas de support valide, utiliser la méthode de pourcentage fixe
                stop_level = self.calculate_fixed_percent(entry_price, side)
                
        else:  # short
            # Pour une position courte, trouver la résistance la plus proche au-dessus du prix d'entrée
            valid_resistances = [r for r in resistance_levels if r > entry_price]
            
            if valid_resistances:
                # Prendre la résistance la plus proche
                stop_level = min(valid_resistances)
            else:
                # Si pas de résistance valide, utiliser la méthode de pourcentage fixe
                stop_level = self.calculate_fixed_percent(entry_price, side)
                
        logger.debug(f"Stop loss S/R calculé: {stop_level} (entrée: {entry_price}, côté: {side})")
        return stop_level
    
    def calculate_trailing_stop(self, entry_price, current_price, highest_price, lowest_price, side, activation_percent=0.01, trail_percent=0.02):
        """
        Calcule un stop loss trailing
        
        Args:
            entry_price (float): Prix d'entrée
            current_price (float): Prix actuel
            highest_price (float): Prix le plus haut depuis l'entrée
            lowest_price (float): Prix le plus bas depuis l'entrée
            side (str): Direction de la position ('long' ou 'short')
            activation_percent (float, optional): Pourcentage d'activation
            trail_percent (float, optional): Pourcentage de trailing
            
        Returns:
            float: Niveau de stop loss
        """
        # Calculer le seuil d'activation
        if side.lower() == 'long':
            activation_threshold = entry_price * (1 + activation_percent)
            
            # Vérifier si le stop trailing est activé
            if highest_price >= activation_threshold:
                # Calculer le stop trailing
                stop_level = highest_price * (1 - trail_percent)
                
                # Ne jamais baisser le stop
                initial_stop = self.calculate_fixed_percent(entry_price, side)
                stop_level = max(stop_level, initial_stop)
            else:
                # Utiliser le stop initial
                stop_level = self.calculate_fixed_percent(entry_price, side)
                
        else:  # short
            activation_threshold = entry_price * (1 - activation_percent)
            
            # Vérifier si le stop trailing est activé
            if lowest_price <= activation_threshold:
                # Calculer le stop trailing
                stop_level = lowest_price * (1 + trail_percent)
                
                # Ne jamais monter le stop
                initial_stop = self.calculate_fixed_percent(entry_price, side)
                stop_level = min(stop_level, initial_stop)
            else:
                # Utiliser le stop initial
                stop_level = self.calculate_fixed_percent(entry_price, side)
                
        logger.debug(f"Stop loss trailing calculé: {stop_level} (entrée: {entry_price}, actuel: {current_price}, côté: {side})")
        return stop_level