"""
Module de visualisation
Fournit des fonctions pour visualiser les résultats des backtests et autres données
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration de style pour matplotlib
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def plot_backtest_results(results, show_trades=True, show_drawdown=True, save_path=None):
    """
    Trace les résultats d'un backtest
    
    Args:
        results (dict): Résultats du backtest
        show_trades (bool, optional): Afficher les trades
        show_drawdown (bool, optional): Afficher le drawdown
        save_path (str, optional): Chemin pour sauvegarder l'image
    """
    try:
        if not results:
            logger.warning("Aucun résultat à afficher")
            return
        
        # Récupérer la courbe d'équité
        equity_curve = results.get('equity_curve')
        if equity_curve is None or equity_curve.empty:
            logger.warning("Courbe d'équité vide")
            return
        
        # Récupérer les trades
        trades = results.get('trades', [])
        
        # Récupérer les signaux
        signals = results.get('signals')
        
        # Symbole et timeframe
        symbol = results.get('symbol', 'unknown')
        timeframe = results.get('timeframe', '')
        
        # Dates
        start_date = results.get('start_date')
        end_date = results.get('end_date')
        if start_date is None and not equity_curve.empty:
            start_date = equity_curve.index[0]
        if end_date is None and not equity_curve.empty:
            end_date = equity_curve.index[-1]
        
        # Statistiques
        stats = {
            'total_return': f"{results.get('profit_loss_pct', 0):.2f}%",
            'sharpe_ratio': f"{results.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': f"{results.get('max_drawdown', 0)*100:.2f}%",
            'win_rate': f"{results.get('win_rate', 0):.2f}%",
            'profit_factor': f"{results.get('profit_factor', 0):.2f}",
            'total_trades': results.get('total_trades', 0),
        }
        
        # Créer la figure
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(14, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Tracer la courbe d'équité
        ax1.plot(equity_curve.index, equity_curve['equity'], label='Equity', linewidth=2)
        
        # Ajouter une ligne horizontale pour le capital initial
        initial_capital = results.get('initial_capital', equity_curve['equity'].iloc[0])
        ax1.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.3, label='Initial Capital')
        
        # Tracer les trades si demandé
        if show_trades and trades:
            # Vérifier si les trades ont le bon format
            valid_trades = True
            for t in trades:
                if not isinstance(t, dict) or 'entry_time' not in t or 'exit_time' not in t:
                    valid_trades = False
                    logger.warning("Format de trades invalide pour la visualisation")
                    break
            
            if valid_trades:
                # Séparer les trades gagnants et perdants
                winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
                
                # Tracer les points d'entrée/sortie des trades gagnants
                if winning_trades:
                    try:
                        entry_times = [t['entry_time'] for t in winning_trades]
                        exit_times = [t['exit_time'] for t in winning_trades]
                        
                        # Trouver les valeurs d'équité correspondantes en toute sécurité
                        entry_values = []
                        exit_values = []
                        
                        for t in winning_trades:
                            # Trouver l'index le plus proche
                            entry_idx = equity_curve.index.get_indexer([t['entry_time']], method='nearest')[0]
                            exit_idx = equity_curve.index.get_indexer([t['exit_time']], method='nearest')[0]
                            
                            if 0 <= entry_idx < len(equity_curve):
                                entry_values.append(equity_curve['equity'].iloc[entry_idx])
                            
                            if 0 <= exit_idx < len(equity_curve):
                                exit_values.append(equity_curve['equity'].iloc[exit_idx])
                        
                        if entry_values and exit_values:
                            ax1.scatter(entry_times, entry_values, color='g', marker='^', s=100, alpha=0.7, label='Win Entry')
                            ax1.scatter(exit_times, exit_values, color='g', marker='o', s=100, alpha=0.7, label='Win Exit')
                    except Exception as e:
                        logger.warning(f"Erreur lors du traçage des trades gagnants: {e}")
                
                # Tracer les points d'entrée/sortie des trades perdants
                if losing_trades:
                    try:
                        entry_times = [t['entry_time'] for t in losing_trades]
                        exit_times = [t['exit_time'] for t in losing_trades]
                        
                        # Trouver les valeurs d'équité correspondantes en toute sécurité
                        entry_values = []
                        exit_values = []
                        
                        for t in losing_trades:
                            # Trouver l'index le plus proche
                            entry_idx = equity_curve.index.get_indexer([t['entry_time']], method='nearest')[0]
                            exit_idx = equity_curve.index.get_indexer([t['exit_time']], method='nearest')[0]
                            
                            if 0 <= entry_idx < len(equity_curve):
                                entry_values.append(equity_curve['equity'].iloc[entry_idx])
                            
                            if 0 <= exit_idx < len(equity_curve):
                                exit_values.append(equity_curve['equity'].iloc[exit_idx])
                        
                        if entry_values and exit_values:
                            ax1.scatter(entry_times, entry_values, color='r', marker='^', s=100, alpha=0.7, label='Loss Entry')
                            ax1.scatter(exit_times, exit_values, color='r', marker='o', s=100, alpha=0.7, label='Loss Exit')
                    except Exception as e:
                        logger.warning(f"Erreur lors du traçage des trades perdants: {e}")
        
        # Tracer le drawdown si demandé
        if show_drawdown:
            try:
                # Calculer le drawdown
                peak = equity_curve['equity'].cummax()
                drawdown = (peak - equity_curve['equity']) / peak * 100
                
                # Tracer le drawdown
                ax2.fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_xlabel('Date')
                ax2.set_title('Drawdown')
                ax2.set_ylim(0, max(drawdown) * 1.1 if len(drawdown) > 0 and drawdown.max() > 0 else 10)
            except Exception as e:
                logger.warning(f"Erreur lors du calcul ou du traçage du drawdown: {e}")
        
        # Configuration du graphique principal
        title = f'Backtest Results - {symbol} {timeframe}'
        if start_date and end_date:
            start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)
            title += f" ({start_str} to {end_str})"
            
        ax1.set_title(title)
        ax1.set_ylabel('Equity')
        if not show_drawdown:
            ax1.set_xlabel('Date')
        
        # Formatter les dates sur l'axe X
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Ajouter une légende
        if show_trades:
            ax1.legend(loc='upper left')
        
        # Ajouter les statistiques dans un cadre
        stats_text = '\n'.join([f"{k}: {v}" for k, v in stats.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder l'image si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure sauvegardée dans {save_path}")
        
        # Afficher la figure
        plt.show()
        
    except Exception as e:
        logger.error(f"Erreur lors du traçage des résultats: {e}")
        import traceback
        logger.error(traceback.format_exc())

def plot_equity_curve(equity_curve, title="Equity Curve", save_path=None):
    """
    Trace une courbe d'équité
    
    Args:
        equity_curve (pd.DataFrame): Courbe d'équité
        title (str, optional): Titre du graphique
        save_path (str, optional): Chemin pour sauvegarder l'image
    """
    plt.figure(figsize=(14, 8))
    plt.plot(equity_curve.index, equity_curve['equity'], linewidth=2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    
    # Formatter les dates sur l'axe X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Ajouter des statistiques
    initial_capital = equity_curve['equity'].iloc[0]
    final_capital = equity_curve['equity'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Calculer le drawdown
    peak = equity_curve['equity'].cummax()
    drawdown = (peak - equity_curve['equity']) / peak * 100
    max_drawdown = drawdown.max()
    
    # Ajouter les statistiques dans un cadre
    stats_text = f"Initial Capital: {initial_capital:.2f}\n"
    stats_text += f"Final Capital: {final_capital:.2f}\n"
    stats_text += f"Total Return: {total_return:.2f}%\n"
    stats_text += f"Max Drawdown: {max_drawdown:.2f}%"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder l'image si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans {save_path}")
        
    # Afficher la figure
    plt.show()

def plot_drawdown(equity_curve, title="Drawdown", save_path=None):
    """
    Trace le drawdown
    
    Args:
        equity_curve (pd.DataFrame): Courbe d'équité
        title (str, optional): Titre du graphique
        save_path (str, optional): Chemin pour sauvegarder l'image
    """
    # Calculer le drawdown
    peak = equity_curve['equity'].cummax()
    drawdown = (peak - equity_curve['equity']) / peak * 100
    
    plt.figure(figsize=(14, 8))
    plt.fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    
    # Formatter les dates sur l'axe X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Ajouter le drawdown maximum
    max_drawdown = drawdown.max()
    max_drawdown_date = drawdown.idxmax()
    
    plt.axhline(y=max_drawdown, color='r', linestyle='--', label=f'Max Drawdown: {max_drawdown:.2f}%')
    plt.annotate(f'{max_drawdown:.2f}%', xy=(max_drawdown_date, max_drawdown),
                xytext=(max_drawdown_date, max_drawdown+5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.legend()
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder l'image si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans {save_path}")
        
    # Afficher la figure
    plt.show()

def plot_monthly_returns(equity_curve, title="Monthly Returns", save_path=None):
    """
    Trace les rendements mensuels
    
    Args:
        equity_curve (pd.DataFrame): Courbe d'équité
        title (str, optional): Titre du graphique
        save_path (str, optional): Chemin pour sauvegarder l'image
    """
    # Calculer les rendements quotidiens
    daily_returns = equity_curve['equity'].pct_change().dropna()
    
    # Regrouper par mois et calculer les rendements mensuels
    monthly_returns = daily_returns.groupby(pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Créer un DataFrame avec année et mois
    returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Pivoter pour avoir les mois en colonnes et les années en lignes
    pivot_df = returns_df.pivot(index='Year', columns='Month', values='Return')
    
    # Remplacer les noms de colonnes par les noms des mois
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_df.columns = [month_names[i-1] for i in pivot_df.columns]
    
    # Tracer la heatmap
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot_df, annot=True, cmap=sns.diverging_palette(10, 220, n=21), fmt='.1f', center=0)
    
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder l'image si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans {save_path}")
        
    # Afficher la figure
    plt.show()
    
    return pivot_df

def plot_trade_analysis(trades, title="Trade Analysis", save_path=None):
    """
    Analyse et trace des statistiques sur les trades
    
    Args:
        trades (list): Liste des trades
        title (str, optional): Titre du graphique
        save_path (str, optional): Chemin pour sauvegarder l'image
    """
    if not trades:
        logger.warning("Aucun trade à analyser")
        return
    
    # Créer un DataFrame des trades
    df_trades = pd.DataFrame(trades)
    
    # Séparer les trades gagnants et perdants
    winning_trades = df_trades[df_trades['pnl'] > 0]
    losing_trades = df_trades[df_trades['pnl'] <= 0]
    
    # Calculer les statistiques
    win_rate = len(winning_trades) / len(df_trades) * 100
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    # Calculer le profit factor
    total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculer la durée des trades
    if 'entry_time' in df_trades.columns and 'exit_time' in df_trades.columns:
        df_trades['duration'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600  # en heures
        avg_duration = df_trades['duration'].mean()
    else:
        avg_duration = None
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution des P&L
    sns.histplot(df_trades['pnl'], bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Distribution des P&L')
    axes[0, 0].set_xlabel('P&L')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Distribution des P&L en %
    if 'pnl_pct' in df_trades.columns:
        sns.histplot(df_trades['pnl_pct'], bins=20, kde=True, ax=axes[0, 1])
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Distribution des P&L (%)')
        axes[0, 1].set_xlabel('P&L (%)')
        axes[0, 1].set_ylabel('Fréquence')
    
    # Distribution des durées des trades
    if avg_duration is not None:
        sns.histplot(df_trades['duration'], bins=20, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution des durées des trades (heures)')
        axes[1, 0].set_xlabel('Durée (heures)')
        axes[1, 0].set_ylabel('Fréquence')
    
    # P&L cumulé
    df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
    axes[1, 1].plot(range(len(df_trades)), df_trades['cumulative_pnl'], marker='o')
    axes[1, 1].set_title('P&L Cumulé')
    axes[1, 1].set_xlabel('Trade #')
    axes[1, 1].set_ylabel('P&L Cumulé')
    axes[1, 1].grid(True)
    
    # Ajouter les statistiques dans un cadre
    stats_text = f"Nombre de trades: {len(df_trades)}\n"
    stats_text += f"Trades gagnants: {len(winning_trades)} ({win_rate:.2f}%)\n"
    stats_text += f"Trades perdants: {len(losing_trades)} ({100-win_rate:.2f}%)\n"
    stats_text += f"Gain moyen: {avg_win:.2f}\n"
    stats_text += f"Perte moyenne: {avg_loss:.2f}\n"
    stats_text += f"Profit Factor: {profit_factor:.2f}\n"
    if avg_duration is not None:
        stats_text += f"Durée moyenne: {avg_duration:.2f} heures"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.95, stats_text, fontsize=12, ha='center', va='top', bbox=props)
    
    fig.suptitle(title, fontsize=16)
    
    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Sauvegarder l'image si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans {save_path}")
        
    # Afficher la figure
    plt.show()