"""
Configuration globale du bot de trading
"""
import os
from datetime import datetime
from pathlib import Path

# Chemin de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemins des dossiers
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")

# Créer les dossiers s'ils n'existent pas
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration des logs
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")

# Configuration de l'API de l'exchange
EXCHANGE = {
    "name": "binance",  # Options: binance, ftx, kraken, etc.
    "api_key": os.environ.get("EXCHANGE_API_KEY", ""),
    "api_secret": os.environ.get("EXCHANGE_API_SECRET", ""),
    "testnet": True,  # Utiliser le testnet/sandbox (recommandé pour les tests)
}

# Paramètres généraux du trading
TRADING = {
    "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],  # Paires à trader
    "timeframe": "15m",  # Timeframe principal (1m, 5m, 15m, 1h, 4h, 1d, etc.)
    "additional_timeframes": ["1h", "4h"],  # Timeframes additionnels pour analyse multi-temporelle
    "base_currency": "USDT",  # Devise de base pour le calcul du capital
}

# Paramètres de gestion du capital
CAPITAL = {
    "initial_capital": 10000,  # Capital initial
    "risk_per_trade": 0.01,  # Risque par trade (1% du capital)
    "max_positions": 5,  # Nombre maximum de positions simultanées
    "max_risk_per_day": 0.05,  # Risque maximum par jour (5% du capital)
    "max_position_size": 0.2,  # Taille max d'une position (20% du capital)
}

# Configuration des notifications
NOTIFICATIONS = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "",
        "receiver_email": "",
        "password": os.environ.get("EMAIL_PASSWORD", ""),
    },
    "telegram": {
        "enabled": False,
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    },
}

# Configuration du backtesting
BACKTEST = {
    "start_date": datetime(2020, 1, 1),
    "end_date": datetime(2023, 1, 1),
    "commission_rate": 0.001,  # 0.1% de frais de commission par trade
    "slippage": 0.0005,  # 0.05% de slippage
}

# Mode d'exécution
EXECUTION_MODE = "backtest"  # Options: "backtest", "paper", "live"