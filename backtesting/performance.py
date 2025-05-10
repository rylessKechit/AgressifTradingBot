"""
Module d'évaluation des performances de trading
Calcule les métriques de performance des stratégies de trading
"""
import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Classe pour analyser les performances d'une stratégie de trading
    """
    
    @staticmethod
    def calculate_metrics(equity_curve, trades, risk_free_rate=0.0):
        """
        Calcule les métriques de performance à partir de la courbe d'équité et des trades
        
        Args:
            equity_curve (pd.DataFrame): Courbe d'équité (index=dates, columns=['equity'])
            trades (list): Liste des trades
            risk_free_rate (float, optional): Taux sans risque annualisé
            
        Returns:
            dict: Métriques de performance
        """
        metrics = {}
        
        # Vérifier les données
        if equity_curve is None or equity_curve.empty:
            logger.warning("Courbe d'équité vide, impossible de calculer les métriques")
            return metrics
            
        # Convertir la liste des trades en DataFrame si nécessaire
        if trades and isinstance(trades, list):
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = trades
        
        # --- Métriques de rentabilité ---
        
        # Capital initial et final
        initial_capital = equity_curve['equity'].iloc[0]
        final_capital = equity_curve['equity'].iloc[-1]
        
        # Profit/Perte total
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        metrics['initial_capital'] = initial_capital
        metrics['final_capital'] = final_capital
        metrics['total_return'] = total_return
        metrics['total_return_pct'] = total_return_pct
        
        # Rendement annualisé
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        
        if years > 0:
            metrics['annualized_return'] = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            metrics['annualized_return'] = 0
        
        # --- Métriques de risque ---
        
        # Calculer les rendements quotidiens
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        
        # Volatilité (écart-type annualisé des rendements quotidiens)
        daily_std = equity_curve['daily_return'].std()
        metrics['volatility'] = daily_std * np.sqrt(252) * 100  # Convertir en % annualisé
        
        # Ratio de Sharpe (rendement / risque)
        if daily_std > 0:
            excess_return = equity_curve['daily_return'].mean() - (risk_free_rate / 252)
            metrics['sharpe_ratio'] = excess_return / daily_std * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Ratio de Sortino (rendement / risque négatif)
        negative_returns = equity_curve['daily_return'][equity_curve['daily_return'] < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            excess_return = equity_curve['daily_return'].mean() - (risk_free_rate / 252)
            metrics['sortino_ratio'] = excess_return / negative_returns.std() * np.sqrt(252)
        else:
            metrics['sortino_ratio'] = 0
        
        # Maximum Drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak'] * 100
        
        metrics['max_drawdown'] = equity_curve['drawdown'].max()
        
        # Ratio de Calmar (rendement annualisé / max drawdown)
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0
        
        # --- Métriques des trades ---
        
        if trades_df is not None and not trades_df.empty:
            # Nombre de trades
            metrics['total_trades'] = len(trades_df)
            
            # Trades gagnants/perdants
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            # Taux de réussite
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
            else:
                metrics['win_rate'] = 0
            
            # Profit moyen des trades gagnants
            if len(winning_trades) > 0:
                metrics['avg_profit'] = winning_trades['pnl'].mean()
                metrics['avg_profit_pct'] = winning_trades['pnl_percent'].mean() if 'pnl_percent' in winning_trades.columns else 0
            else:
                metrics['avg_profit'] = 0
                metrics['avg_profit_pct'] = 0
            
            # Perte moyenne des trades perdants
            if len(losing_trades) > 0:
                metrics['avg_loss'] = losing_trades['pnl'].mean()
                metrics['avg_loss_pct'] = losing_trades['pnl_percent'].mean() if 'pnl_percent' in losing_trades.columns else 0
            else:
                metrics['avg_loss'] = 0
                metrics['avg_loss_pct'] = 0
            
            # Ratio profit/perte (profit factor)
            total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            
            if total_loss > 0:
                metrics['profit_factor'] = total_profit / total_loss
            else:
                metrics['profit_factor'] = float('inf') if total_profit > 0 else 0
            
            # Espérance mathématique (expectancy)
            if metrics['total_trades'] > 0:
                metrics['expectancy'] = (metrics['win_rate'] / 100 * metrics['avg_profit'] + 
                                       (1 - metrics['win_rate'] / 100) * metrics['avg_loss'])
                
                # Ratio Gain/Perte (reward/risk ratio)
                if metrics['avg_loss'] != 0:
                    metrics['reward_risk_ratio'] = abs(metrics['avg_profit'] / metrics['avg_loss'])
                else:
                    metrics['reward_risk_ratio'] = float('inf')
            else:
                metrics['expectancy'] = 0
                metrics['reward_risk_ratio'] = 0
            
            # Nombre de trades consécutifs gagnants/perdants
            if len(trades_df) > 0:
                # Ajouter une colonne indiquant si le trade est gagnant
                trades_df['is_winner'] = trades_df['pnl'] > 0
                
                # Initialiser les compteurs
                current_streak = 1
                max_win_streak = 0
                max_loss_streak = 0
                
                # Parcourir les trades (à partir du deuxième)
                for i in range(1, len(trades_df)):
                    # Si le trade actuel a le même résultat que le précédent
                    if trades_df['is_winner'].iloc[i] == trades_df['is_winner'].iloc[i-1]:
                        current_streak += 1
                    else:
                        # Réinitialiser la série
                        current_streak = 1
                    
                    # Mettre à jour les séries maximales
                    if trades_df['is_winner'].iloc[i]:
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                
                metrics['max_consecutive_wins'] = max_win_streak
                metrics['max_consecutive_losses'] = max_loss_streak
            else:
                metrics['max_consecutive_wins'] = 0
                metrics['max_consecutive_losses'] = 0
        
        # --- Métriques statistiques avancées ---
        
        # Coefficient de corrélation entre l'équité et le temps (tendance linéaire)
        try:
            time_ix = np.arange(len(equity_curve))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_ix, equity_curve['equity'])
            metrics['equity_curve_correlation'] = r_value
            metrics['equity_curve_slope'] = slope
        except:
            metrics['equity_curve_correlation'] = 0
            metrics['equity_curve_slope'] = 0
        
        # Z-score (écart par rapport à la moyenne en unités d'écart-type)
        if metrics['annualized_return'] != 0 and metrics['volatility'] > 0:
            metrics['z_score'] = metrics['annualized_return'] / metrics['volatility']
        else:
            metrics['z_score'] = 0
        
        # Skewness et Kurtosis (asymétrie et aplatissement de la distribution des rendements)
        if len(equity_curve['daily_return'].dropna()) > 0:
            metrics['skewness'] = equity_curve['daily_return'].skew()
            metrics['kurtosis'] = equity_curve['daily_return'].kurt()
        else:
            metrics['skewness'] = 0
            metrics['kurtosis'] = 0
        
        # Value at Risk (VaR) - perte maximale avec une probabilité donnée
        if len(equity_curve['daily_return'].dropna()) > 0:
            metrics['var_95'] = np.percentile(equity_curve['daily_return'], 5) * 100  # 95% VaR (en %)
            metrics['var_99'] = np.percentile(equity_curve['daily_return'], 1) * 100  # 99% VaR (en %)
        else:
            metrics['var_95'] = 0
            metrics['var_99'] = 0
        
        # Durée moyenne des trades
        if trades_df is not None and not trades_df.empty and 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # en heures
            metrics['avg_trade_duration'] = trades_df['duration'].mean()
            metrics['max_trade_duration'] = trades_df['duration'].max()
            metrics['min_trade_duration'] = trades_df['duration'].min()
        else:
            metrics['avg_trade_duration'] = 0
            metrics['max_trade_duration'] = 0
            metrics['min_trade_duration'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_trade_statistics(trades):
        """
        Calcule des statistiques détaillées sur les trades
        
        Args:
            trades (list): Liste des trades
            
        Returns:
            dict: Statistiques des trades
        """
        stats = {}
        
        # Vérifier les données
        if not trades:
            logger.warning("Liste des trades vide, impossible de calculer les statistiques")
            return stats
            
        # Convertir en DataFrame si nécessaire
        if isinstance(trades, list):
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = trades
        
        if trades_df.empty:
            return stats
        
        # Nombre total de trades
        stats['total_trades'] = len(trades_df)
        
        # Trades par jour
        if 'entry_time' in trades_df.columns:
            start_date = trades_df['entry_time'].min()
            end_date = trades_df['exit_time'].max() if 'exit_time' in trades_df.columns else trades_df['entry_time'].max()
            days = (end_date - start_date).days
            
            if days > 0:
                stats['trades_per_day'] = stats['total_trades'] / days
            else:
                stats['trades_per_day'] = stats['total_trades']
        else:
            stats['trades_per_day'] = 0
        
        # Répartition des trades par résultat
        if 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
                
                # Profit total
                stats['total_profit'] = trades_df['pnl'].sum()
                
                # Profit moyen par trade
                stats['avg_profit_per_trade'] = stats['total_profit'] / stats['total_trades']
                
                # Profit moyen des gagnants / Perte moyenne des perdants
                if stats['winning_trades'] > 0:
                    stats['avg_win'] = winning_trades['pnl'].mean()
                else:
                    stats['avg_win'] = 0
                    
                if stats['losing_trades'] > 0:
                    stats['avg_loss'] = losing_trades['pnl'].mean()
                else:
                    stats['avg_loss'] = 0
                
                # Ratio risque/récompense
                if stats['avg_loss'] != 0:
                    stats['risk_reward_ratio'] = abs(stats['avg_win'] / stats['avg_loss'])
                else:
                    stats['risk_reward_ratio'] = float('inf')
                
                # Profit factor
                total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
                total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
                
                if total_loss > 0:
                    stats['profit_factor'] = total_profit / total_loss
                else:
                    stats['profit_factor'] = float('inf') if total_profit > 0 else 0
                
                # Espérance mathématique
                stats['expectancy'] = (stats['win_rate'] / 100 * stats['avg_win'] + 
                                      (1 - stats['win_rate'] / 100) * stats['avg_loss'])
        
        # Répartition par direction
        if 'side' in trades_df.columns:
            long_trades = trades_df[trades_df['side'] == 'long']
            short_trades = trades_df[trades_df['side'] == 'short']
            
            stats['long_trades'] = len(long_trades)
            stats['short_trades'] = len(short_trades)
            
            if len(long_trades) > 0:
                stats['long_win_rate'] = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades)) * 100
                stats['long_profit'] = long_trades['pnl'].sum()
            else:
                stats['long_win_rate'] = 0
                stats['long_profit'] = 0
                
            if len(short_trades) > 0:
                stats['short_win_rate'] = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades)) * 100
                stats['short_profit'] = short_trades['pnl'].sum()
            else:
                stats['short_win_rate'] = 0
                stats['short_profit'] = 0
        
        # Répartition temporelle
        if 'entry_time' in trades_df.columns:
            # Par mois
            trades_df['month'] = trades_df['entry_time'].dt.month
            monthly_trades = trades_df.groupby('month').size()
            monthly_profit = trades_df.groupby('month')['pnl'].sum() if 'pnl' in trades_df.columns else None
            
            stats['monthly_trades'] = monthly_trades.to_dict()
            
            if monthly_profit is not None:
                stats['monthly_profit'] = monthly_profit.to_dict()
            
            # Par jour de la semaine
            trades_df['day_of_week'] = trades_df['entry_time'].dt.dayofweek
            daily_trades = trades_df.groupby('day_of_week').size()
            daily_profit = trades_df.groupby('day_of_week')['pnl'].sum() if 'pnl' in trades_df.columns else None
            
            # Convertir les jours de la semaine en noms
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            stats['daily_trades'] = {day_names[day]: count for day, count in daily_trades.items()}
            
            if daily_profit is not None:
                stats['daily_profit'] = {day_names[day]: profit for day, profit in daily_profit.items()}
            
            # Par heure du jour
            trades_df['hour_of_day'] = trades_df['entry_time'].dt.hour
            hourly_trades = trades_df.groupby('hour_of_day').size()
            hourly_profit = trades_df.groupby('hour_of_day')['pnl'].sum() if 'pnl' in trades_df.columns else None
            
            stats['hourly_trades'] = hourly_trades.to_dict()
            
            if hourly_profit is not None:
                stats['hourly_profit'] = hourly_profit.to_dict()
        
        # Durée des trades
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # en heures
            
            stats['avg_duration'] = trades_df['duration'].mean()
            stats['min_duration'] = trades_df['duration'].min()
            stats['max_duration'] = trades_df['duration'].max()
            stats['median_duration'] = trades_df['duration'].median()
            
            # Corrélation durée / profit
            if 'pnl' in trades_df.columns:
                stats['duration_profit_correlation'] = trades_df[['duration', 'pnl']].corr().iloc[0, 1]
        
        # Répartition par raison de sortie
        if 'reason' in trades_df.columns:
            reason_counts = trades_df['reason'].value_counts()
            stats['exit_reasons'] = reason_counts.to_dict()
            
            # Profit moyen par raison de sortie
            if 'pnl' in trades_df.columns:
                reason_profits = trades_df.groupby('reason')['pnl'].mean()
                stats['reason_avg_profits'] = reason_profits.to_dict()
        
        return stats
    
    @staticmethod
    def calculate_monthly_returns(equity_curve):
        """
        Calcule les rendements mensuels à partir de la courbe d'équité
        
        Args:
            equity_curve (pd.DataFrame): Courbe d'équité (index=dates, columns=['equity'])
            
        Returns:
            pd.DataFrame: Rendements mensuels (années en index, mois en colonnes)
        """
        if equity_curve is None or equity_curve.empty:
            logger.warning("Courbe d'équité vide, impossible de calculer les rendements mensuels")
            return pd.DataFrame()
        
        # Calculer les rendements quotidiens
        equity_curve = equity_curve.copy()
        equity_curve['daily_returns'] = equity_curve['equity'].pct_change()
        
        # Calculer les rendements mensuels
        monthly_returns = equity_curve['daily_returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Créer un DataFrame avec année et mois
        returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100  # En pourcentage
        })
        
        # Pivoter pour avoir les mois en colonnes et les années en lignes
        pivot_df = returns_df.pivot(index='Year', columns='Month', values='Return')
        
        # Remplacer les noms de colonnes par les noms des mois
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_df.columns = [month_names[i-1] for i in pivot_df.columns]
        
        return pivot_df
    
    @staticmethod
    def compare_strategies(results_list, names=None):
        """
        Compare les performances de plusieurs stratégies
        
        Args:
            results_list (list): Liste des résultats de backtest
            names (list, optional): Liste des noms des stratégies
            
        Returns:
            pd.DataFrame: Comparaison des métriques de performance
        """
        if not results_list:
            logger.warning("Liste des résultats vide, impossible de comparer les stratégies")
            return pd.DataFrame()
            
        if names is None:
            names = [f"Strategy {i+1}" for i in range(len(results_list))]
            
        # Métriques à comparer
        metrics_to_compare = [
            'total_return_pct',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'total_trades',
            'expectancy'
        ]
        
        # Créer le DataFrame de comparaison
        comparison = pd.DataFrame(index=metrics_to_compare, columns=names)
        
        # Remplir le DataFrame
        for i, results in enumerate(results_list):
            for metric in metrics_to_compare:
                if metric in results:
                    comparison.loc[metric, names[i]] = results[metric]
                else:
                    comparison.loc[metric, names[i]] = None
        
        return comparison


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de courbe d'équité
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Générer des données d'exemple
    dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='D')
    np.random.seed(42)
    
    # Simuler une courbe d'équité
    initial_capital = 10000
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))
    equity = initial_capital * (1 + daily_returns).cumprod()
    
    equity_curve = pd.DataFrame({
        'equity': equity
    }, index=dates)
    
    # Simuler des trades
    n_trades = 50
    trade_dates = np.sort(np.random.choice(dates, n_trades * 2, replace=False))
    
    trades = []
    for i in range(0, len(trade_dates), 2):
        if i + 1 < len(trade_dates):
            entry_time = trade_dates[i]
            exit_time = trade_dates[i + 1]
            
            entry_price = equity_curve.loc[entry_time, 'equity'] / 100
            exit_price = equity_curve.loc[exit_time, 'equity'] / 100
            
            size = np.random.uniform(0.1, 1.0)
            side = np.random.choice(['long', 'short'])
            
            if side == 'long':
                pnl = (exit_price - entry_price) * size
            else:
                pnl = (entry_price - exit_price) * size
                
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'side': side,
                'pnl': pnl,
                'pnl_percent': pnl / (entry_price * size) * 100,
                'reason': np.random.choice(['Signal', 'Take Profit', 'Stop Loss'])
            })
    
    # Calculer les métriques de performance
    metrics = PerformanceAnalyzer.calculate_metrics(equity_curve, trades)
    
    # Afficher les métriques
    print("Métriques de performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Calculer les statistiques des trades
    trade_stats = PerformanceAnalyzer.calculate_trade_statistics(trades)
    
    # Afficher les statistiques
    print("\nStatistiques des trades:")
    for key, value in trade_stats.items():
        print(f"{key}: {value}")
    
    # Calculer les rendements mensuels
    monthly_returns = PerformanceAnalyzer.calculate_monthly_returns(equity_curve)
    
    # Afficher les rendements mensuels
    print("\nRendements mensuels:")
    print(monthly_returns)
    
    # Simuler plusieurs stratégies
    strategies_results = []
    for _ in range(3):
        daily_returns = np.random.normal(0.0005 + np.random.uniform(-0.0002, 0.0002),
                                         0.01 + np.random.uniform(-0.002, 0.002),
                                         len(dates))
        equity = initial_capital * (1 + daily_returns).cumprod()
        
        equity_curve = pd.DataFrame({
            'equity': equity
        }, index=dates)
        
        metrics = PerformanceAnalyzer.calculate_metrics(equity_curve, trades)
        strategies_results.append(metrics)
    
    # Comparer les stratégies
    comparison = PerformanceAnalyzer.compare_strategies(
        strategies_results,
        names=['Strategy A', 'Strategy B', 'Strategy C']
    )
    
    # Afficher la comparaison
    print("\nComparaison des stratégies:")
    print(comparison)