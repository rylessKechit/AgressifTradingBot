"""
Paramètres des stratégies de trading
Ce fichier contient les paramètres spécifiques pour chaque stratégie
"""

# Paramètres pour la stratégie de suivi de tendance
TREND_FOLLOWING_PARAMS = {
    # Moyennes mobiles
    "fast_ema": 8,
    "medium_ema": 21,
    "slow_ema": 50,
    "very_slow_ema": 200,
    
    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # ADX (Average Directional Index) pour la force de tendance
    "adx_period": 14,
    "adx_threshold": 25,  # Considéré comme tendance forte si > 25
    
    # Filtres de tendance
    "trend_filter_timeframe": "4h",  # Timeframe utilisé pour le filtre de tendance
    "trend_filter_period": 100,  # Nombre de périodes pour évaluer la tendance
    
    # Seuils de signaux
    "buy_threshold": 0.7,  # Seuil pour générer un signal d'achat (0-1)
    "sell_threshold": -0.7,  # Seuil pour générer un signal de vente (-1-0)
    
    # Poids dans la stratégie combinée
    "weight": 0.4,  # 40% du signal final dans la stratégie combinée
}

# Paramètres pour la stratégie de retour à la moyenne
MEAN_REVERSION_PARAMS = {
    # RSI (Relative Strength Index)
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    
    # Bandes de Bollinger
    "bb_period": 20,
    "bb_std": 2.0,  # Nombre d'écarts-types
    
    # Stochastique
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    
    # CCI (Commodity Channel Index)
    "cci_period": 20,
    "cci_overbought": 100,
    "cci_oversold": -100,
    
    # Filtres de volatilité
    "use_volatility_filter": True,
    "atr_period": 14,
    "atr_multiplier": 2.5,
    
    # Seuils de signaux
    "buy_threshold": 0.7,
    "sell_threshold": -0.7,
    
    # Poids dans la stratégie combinée
    "weight": 0.4,  # 40% du signal final dans la stratégie combinée
}

# Paramètres pour la stratégie d'arbitrage
ARBITRAGE_PARAMS = {
    # Exchanges à surveiller pour l'arbitrage
    "exchanges": ["binance", "ftx", "kraken"],
    
    # Seuil minimum de différence de prix pour l'arbitrage
    "min_spread_pct": 0.5,  # 0.5% d'écart minimum
    
    # Temps maximum d'exécution pour une transaction d'arbitrage
    "max_execution_time_seconds": 5,
    
    # Volume minimum pour l'arbitrage
    "min_volume_usd": 10000,  # Volume minimum en USD
    
    # Frais estimés par exchange (en %)
    "exchange_fees": {
        "binance": 0.1,
        "ftx": 0.07,
        "kraken": 0.16,
    },
    
    # Poids dans la stratégie combinée
    "weight": 0.2,  # 20% du signal final dans la stratégie combinée
}

# Paramètres pour la stratégie combinée
COMBINED_STRATEGY_PARAMS = {
    # Seuils de signaux finaux
    "final_buy_threshold": 0.5,
    "final_sell_threshold": -0.5,
    
    # Ajustement des poids selon la volatilité du marché
    "adjust_weights_by_volatility": True,
    
    # Filtres supplémentaires
    "use_volume_filter": True,
    "volume_threshold": 1.5,  # Volume > 1.5x volume moyen
    
    # Paramètres de confirmation multi-timeframes
    "use_multi_timeframe": True,
    "timeframe_alignment_required": False,  # Si True, toutes les timeframes doivent être alignées
}

# Paramètres de gestion des risques et des positions
RISK_MANAGEMENT_PARAMS = {
    # Stop Loss
    "use_fixed_stop_loss": True,
    "fixed_stop_loss_pct": 0.03,  # 3% de stop loss fixe
    
    "use_atr_stop_loss": True,
    "atr_stop_loss_multiplier": 2.5,
    
    "use_trailing_stop": True,
    "trailing_stop_activation_pct": 0.015,  # 1.5% de profit pour activer
    "trailing_stop_distance_pct": 0.01,  # 1% de distance
    
    # Take Profit
    "use_take_profit": True,
    "risk_reward_ratio": 2.0,  # Ratio de risk/reward pour le take profit
    
    "use_trailing_take_profit": True,
    "trailing_take_profit_activation_pct": 0.03,  # 3% de profit pour activer
    
    # Taille des positions
    "position_sizing_method": "risk_based",  # risk_based, fixed, percentage
    "use_volatility_adjustment": True,  # Ajuster selon la volatilité
    "max_position_increase": 2.0,  # Maximum 2x la taille standard en cas de faible volatilité
    "min_position_decrease": 0.5,  # Minimum 0.5x la taille standard en cas de forte volatilité
}