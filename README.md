# Trading Bot

Un bot de trading algorithmique puissant et modulaire, capable d'exécuter différentes stratégies sur les marchés financiers.

## Caractéristiques

- Architecture modulaire et extensible
- Plusieurs stratégies de trading intégrées (tendance, retour à la moyenne, combinées)
- Gestion avancée des risques et du capital
- Backtesting complet avec optimisation des paramètres
- Support pour différentes plateformes d'échange (Binance, FTX, Kraken, etc.)
- Notifications par email des événements importants
- Visualisation détaillée des performances

## Architecture

```
trading-bot/
│
├── config/                      # Configuration du bot
│   ├── __init__.py
│   ├── settings.py              # Paramètres globaux
│   └── strategy_params.py       # Paramètres des stratégies
│
├── data/                        # Gestion des données
│   ├── __init__.py
│   ├── fetcher.py               # Récupération des données
│   ├── processor.py             # Traitement des données
│   └── indicators.py            # Calcul des indicateurs techniques
│
├── strategies/                  # Stratégies de trading
│   ├── __init__.py
│   ├── base_strategy.py         # Classe de base pour les stratégies
│   ├── trend_following.py       # Stratégie de suivi de tendance
│   ├── mean_reversion.py        # Stratégie de retour à la moyenne
│   ├── arbitrage.py             # Stratégie d'arbitrage
│   └── combined_strategy.py     # Stratégie combinée
│
├── risk/                        # Gestion des risques
│   ├── __init__.py
│   ├── position_sizing.py       # Calcul de la taille des positions
│   ├── stop_loss.py             # Stratégies de stop loss
│   └── portfolio.py             # Gestion du portefeuille
│
├── execution/                   # Exécution des ordres
│   ├── __init__.py
│   ├── exchange.py              # Interface avec les exchanges
│   ├── order.py                 # Gestion des ordres
│   └── trader.py                # Exécution des trades
│
├── backtesting/                 # Modules de backtesting
│   ├── __init__.py
│   ├── engine.py                # Moteur de backtesting
│   ├── optimizer.py             # Optimisation des paramètres
│   └── performance.py           # Mesure des performances
│
├── utils/                       # Utilitaires
│   ├── __init__.py
│   ├── logger.py                # Journalisation
│   ├── email_notifier.py        # Notifications par email
│   └── visualizer.py            # Visualisation des résultats
│
├── logs/                        # Dossier pour les logs
│
├── tests/                       # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_strategies.py
│   └── test_execution.py
│
├── main.py                      # Point d'entrée principal
├── backtest.py                  # Script de backtesting
└── requirements.txt             # Dépendances du projet
```

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/votre-utilisateur/trading-bot.git
   cd trading-bot
   ```

2. Installer les dépendances :

   ```bash
   pip3 install -r requirements.txt
   ```

3. Configurer le bot dans le dossier `config/` :
   - Modifier `settings.py` pour les paramètres généraux
   - Modifier `strategy_params.py` pour les paramètres des stratégies

## Utilisation

### Mode Backtesting

Pour tester une stratégie sur des données historiques :

```bash
python backtest.py --strategy combined --symbol BTC/USDT --timeframe 1h --start 2020-01-01 --end 2021-01-01
```

Options disponibles :

- `--strategy` : Stratégie à tester (trend, mean_reversion, combined)
- `--symbol` : Symbole de la paire (BTC/USDT, ETH/USDT, etc.)
- `--timeframe` : Intervalle de temps (1m, 15m, 1h, 4h, 1d, etc.)
- `--start` : Date de début (format YYYY-MM-DD)
- `--end` : Date de fin (format YYYY-MM-DD)
- `--capital` : Capital initial
- `--commission` : Taux de commission
- `--slippage` : Slippage
- `--optimize` : Activer l'optimisation des paramètres

### Mode Trading en Direct

Pour exécuter le bot en mode trading en direct :

```bash
python main.py
```

## Stratégies de Trading

### Suivi de Tendance (Trend Following)

Stratégie qui suit la tendance du marché en utilisant des moyennes mobiles, MACD et autres indicateurs de tendance.

### Retour à la Moyenne (Mean Reversion)

Stratégie qui exploite le phénomène de retour à la moyenne après des mouvements extrêmes, en utilisant RSI, Bandes de Bollinger et autres oscillateurs.

### Stratégie Combinée

Stratégie qui combine plusieurs approches pour améliorer la robustesse et s'adapter à différentes conditions de marché.

## Gestion des Risques

Le bot implémente plusieurs mécanismes de gestion des risques :

- Taille de position dynamique basée sur la volatilité
- Stop loss et take profit automatiques
- Stop loss trailing
- Limites de perte quotidienne
- Diversification entre plusieurs paires

## Personnalisation

Vous pouvez personnaliser le bot en :

- Ajoutant de nouvelles stratégies dans le dossier `strategies/`
- Modifiant les paramètres dans `config/strategy_params.py`
- Ajoutant de nouveaux indicateurs techniques dans `data/indicators.py`
- Implémentant de nouvelles méthodes de gestion des risques dans `risk/`

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Avertissement

Ce logiciel est fourni à titre éducatif et informatif uniquement. Le trading comporte des risques importants et vous pouvez perdre une partie ou la totalité de votre capital. L'auteur ne peut être tenu responsable de vos pertes financières.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
