# Auto-optimisation du Bot de Trading

Ce document explique comment utiliser les fonctionnalités d'auto-optimisation intégrées au bot de trading.

## Introduction

L'auto-optimisation est un processus qui permet au bot de trading d'améliorer automatiquement ses performances en ajustant ses paramètres en fonction des données historiques. Le bot utilise des techniques avancées comme la recherche en grille et la recherche aléatoire pour trouver les meilleurs paramètres pour chaque stratégie de trading.

## Fichiers ajoutés

Les fichiers suivants ont été ajoutés au projet pour permettre l'auto-optimisation :

1. **backtesting/optimizer.py** : Module d'optimisation des stratégies
2. **backtesting/performance.py** : Module d'évaluation des performances
3. **auto_optimizer.py** : Script principal d'auto-optimisation

## Comment utiliser l'auto-optimisation

### 1. Optimisation manuelle

Pour lancer une optimisation manuelle d'une stratégie, exécutez :

```bash
python auto_optimizer.py --strategy combined --symbol BTC/USDT --timeframe 1h --lookback 180 --method grid --jobs 4
```

Options disponibles :

- `--strategy` : Stratégie à optimiser (trend, mean_reversion, combined, all)
- `--symbol` : Symbole de la paire à trader
- `--timeframe` : Intervalle de temps (1m, 15m, 1h, 4h, 1d)
- `--lookback` : Nombre de jours de données historiques à utiliser
- `--method` : Méthode d'optimisation (grid, random)
- `--iterations` : Nombre d'itérations pour la recherche aléatoire
- `--jobs` : Nombre de processus parallèles pour l'optimisation

### 2. Optimisation automatique périodique

Pour planifier une optimisation périodique, utilisez l'option `--schedule` :

```bash
python auto_optimizer.py --strategy combined --symbol BTC/USDT --timeframe 1h --lookback 180 --method grid --schedule --frequency weekly
```

Options supplémentaires :

- `--frequency` : Fréquence d'optimisation (daily, weekly, monthly)

Le script générera une ligne à ajouter au crontab pour planifier l'optimisation périodique.

### 3. Optimisation de toutes les stratégies

Pour optimiser toutes les stratégies disponibles en une seule fois :

```bash
python auto_optimizer.py --strategy all --symbol BTC/USDT --timeframe 1h --lookback 180 --method grid --jobs 4
```

## Fonctionnement

### Processus d'optimisation

1. **Récupération des données historiques** : Le bot récupère les données historiques pour la période spécifiée.
2. **Ajout des indicateurs techniques** : Les indicateurs techniques sont calculés pour les données.
3. **Génération de combinaisons de paramètres** : Le bot génère différentes combinaisons de paramètres à tester.
4. **Backtesting de chaque combinaison** : Chaque combinaison de paramètres est testée sur les données historiques.
5. **Évaluation des performances** : Les performances de chaque combinaison sont évaluées selon plusieurs métriques.
6. **Sélection des meilleurs paramètres** : Les paramètres offrant les meilleures performances sont sélectionnés.
7. **Mise à jour de la configuration** : Les meilleurs paramètres sont automatiquement mis à jour dans le fichier de configuration.

### Métriques d'évaluation

Les combinaisons de paramètres sont évaluées selon plusieurs métriques :

- **Rendement total** : Gain ou perte total sur la période
- **Ratio de Sharpe** : Mesure du rendement ajusté au risque
- **Drawdown maximum** : Perte maximale depuis un sommet
- **Taux de réussite** : Pourcentage de trades gagnants
- **Nombre de trades** : Nombre total de transactions

## Techniques d'optimisation avancées

### Walk-Forward Optimization

Cette technique divise les données historiques en plusieurs périodes successives. Pour chaque période, l'optimisation est effectuée sur une fenêtre glissante, puis les paramètres optimaux sont testés sur la période suivante. Cette approche aide à éviter le surajustement (overfitting).

Pour utiliser cette technique, vous pouvez modifier le code de `StrategyOptimizer` dans `backtesting/optimizer.py`.

### Cross-Validation

Cette technique divise les données en plusieurs sous-ensembles, puis effectue l'optimisation sur certains sous-ensembles et teste les performances sur les autres. Cette approche permet d'évaluer la robustesse des paramètres.

## Conseils pour une optimisation efficace

1. **Évitez le surajustement** : N'optimisez pas trop de paramètres à la fois et utilisez des techniques comme le walk-forward.
2. **Testez sur différentes périodes** : Assurez-vous que les paramètres sont robustes sur différentes conditions de marché.
3. **Limitez la fréquence d'optimisation** : Une optimisation trop fréquente peut conduire à des changements constants de paramètres.
4. **Surveillez les performances en temps réel** : Comparez les performances réelles avec les performances attendues.
5. **Utilisez un capital de test limité** : Commencez avec un petit capital jusqu'à ce que vous soyez confiant dans les paramètres optimisés.

## Exemples de grilles de paramètres

### Stratégie de suivi de tendance

```python
param_grid = {
    'fast_ema': [5, 8, 10, 12, 15],
    'medium_ema': [15, 20, 25, 30, 35],
    'slow_ema': [40, 50, 60, 70, 80],
    'adx_threshold': [15, 20, 25, 30, 35],
    'buy_threshold': [0.3, 0.5, 0.7, 0.9],
    'sell_threshold': [-0.9, -0.7, -0.5, -0.3]
}
```

### Stratégie de retour à la moyenne

```python
param_grid = {
    'rsi_period': [7, 10, 14, 21],
    'rsi_overbought': [65, 70, 75, 80],
    'rsi_oversold': [20, 25, 30, 35],
    'bb_period': [15, 20, 25, 30],
    'bb_std': [1.5, 2.0, 2.5, 3.0]
}
```

### Stratégie combinée

```python
param_grid = {
    'final_buy_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
    'final_sell_threshold': [-0.7, -0.6, -0.5, -0.4, -0.3],
    'adjust_weights_by_volatility': [True, False],
    'use_volume_filter': [True, False]
}
```
