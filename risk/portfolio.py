"""
Module de gestion de portefeuille
Gère l'allocation du capital et le suivi des performances du portefeuille
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Classe pour gérer un portefeuille d'actifs et suivre ses performances
    """
    
    def __init__(self, initial_capital=10000, max_positions=5, max_risk_per_portfolio=0.05):
        """
        Initialise le gestionnaire de portefeuille
        
        Args:
            initial_capital (float, optional): Capital initial
            max_positions (int, optional): Nombre maximum de positions simultanées
            max_risk_per_portfolio (float, optional): Risque maximum pour l'ensemble du portefeuille (en %)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.max_risk_per_portfolio = max_risk_per_portfolio
        
        # Positions actuelles
        self.positions = {}  # {symbol: {"side": "long/short", "size": size, "entry_price": price, "entry_time": time}}
        
        # Historique des transactions
        self.transactions = []
        
        # Courbe d'équité
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])
        self.equity_curve.loc[0] = [datetime.now(), initial_capital]
        self.equity_curve.set_index('timestamp', inplace=True)
        
        logger.info(f"PortfolioManager initialisé avec capital={initial_capital}, max_positions={max_positions}")
    
    def add_position(self, symbol, side, size, entry_price, entry_time=None):
        """
        Ajoute une nouvelle position au portefeuille
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Direction ('long' ou 'short')
            size (float): Taille de la position
            entry_price (float): Prix d'entrée
            entry_time (datetime, optional): Horodatage de l'entrée
            
        Returns:
            bool: True si la position a été ajoutée, False sinon
        """
        # Vérifier si le nombre maximum de positions est atteint
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            logger.warning(f"Nombre maximum de positions atteint ({self.max_positions}), impossible d'ajouter {symbol}")
            return False
            
        # Vérifier si une position existe déjà pour ce symbole
        if symbol in self.positions:
            logger.warning(f"Position existante pour {symbol}, mise à jour au lieu d'ajout")
            self.positions[symbol]["size"] += size
            return True
            
        # Ajouter la position
        self.positions[symbol] = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "entry_time": entry_time or datetime.now(),
            "current_price": entry_price,
            "unrealized_pnl": 0,
            "unrealized_pnl_pct": 0
        }
        
        # Mise à jour du capital (simulée pour paper trading)
        position_value = size * entry_price
        
        # Ajouter la transaction à l'historique
        self.transactions.append({
            "type": "open",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": entry_price,
            "value": position_value,
            "timestamp": entry_time or datetime.now(),
            "remaining_capital": self.current_capital
        })
        
        logger.info(f"Position ajoutée: {side} {size} {symbol} @ {entry_price}")
        return True
    
    def close_position(self, symbol, exit_price, exit_time=None, reason=None):
        """
        Ferme une position existante
        
        Args:
            symbol (str): Symbole de la paire
            exit_price (float): Prix de sortie
            exit_time (datetime, optional): Horodatage de la sortie
            reason (str, optional): Raison de la sortie
            
        Returns:
            dict: Informations sur la position fermée
        """
        if symbol not in self.positions:
            logger.warning(f"Pas de position ouverte pour {symbol}")
            return None
            
        # Récupérer la position
        position = self.positions[symbol]
        
        # Calculer le P&L
        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:  # short
            pnl = (position["entry_price"] - exit_price) * position["size"]
        
        # Calculer le P&L en pourcentage
        position_value = position["entry_price"] * position["size"]
        pnl_pct = (pnl / position_value) * 100
        
        # Mettre à jour le capital
        self.current_capital += pnl
        
        # Ajouter la transaction à l'historique
        self.transactions.append({
            "type": "close",
            "symbol": symbol,
            "side": position["side"],
            "size": position["size"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "timestamp": exit_time or datetime.now(),
            "reason": reason,
            "capital": self.current_capital
        })
        
        # Mettre à jour la courbe d'équité
        self.update_equity_curve(exit_time or datetime.now())
        
        # Information sur la position fermée
        closed_position = {
            "symbol": symbol,
            "side": position["side"],
            "size": position["size"],
            "entry_price": position["entry_price"],
            "entry_time": position["entry_time"],
            "exit_price": exit_price,
            "exit_time": exit_time or datetime.now(),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason
        }
        
        # Supprimer la position du portefeuille
        del self.positions[symbol]
        
        logger.info(f"Position fermée: {position['side']} {position['size']} {symbol} @ {exit_price}, P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
        return closed_position
    
    def update_position(self, symbol, current_price):
        """
        Met à jour une position avec le prix actuel
        
        Args:
            symbol (str): Symbole de la paire
            current_price (float): Prix actuel
            
        Returns:
            dict: Informations sur la position mise à jour
        """
        if symbol not in self.positions:
            return None
            
        # Récupérer la position
        position = self.positions[symbol]
        
        # Mettre à jour le prix actuel
        position["current_price"] = current_price
        
        # Calculer le P&L non réalisé
        if position["side"] == "long":
            unrealized_pnl = (current_price - position["entry_price"]) * position["size"]
        else:  # short
            unrealized_pnl = (position["entry_price"] - current_price) * position["size"]
        
        # Calculer le P&L en pourcentage
        position_value = position["entry_price"] * position["size"]
        unrealized_pnl_pct = (unrealized_pnl / position_value) * 100
        
        # Mettre à jour la position
        position["unrealized_pnl"] = unrealized_pnl
        position["unrealized_pnl_pct"] = unrealized_pnl_pct
        
        return position
    
    def update_positions(self, prices):
        """
        Met à jour toutes les positions avec les prix actuels
        
        Args:
            prices (dict): Prix actuels {symbol: price}
        """
        # Pour chaque position
        for symbol in list(self.positions.keys()):
            if symbol in prices:
                self.update_position(symbol, prices[symbol])
    
    def update_equity_curve(self, timestamp):
        """
        Met à jour la courbe d'équité
        
        Args:
            timestamp (datetime): Horodatage
        """
        # Calculer la valeur actuelle du portefeuille
        portfolio_value = self.current_capital
        
        # Ajouter les P&L non réalisés
        for symbol, position in self.positions.items():
            portfolio_value += position.get("unrealized_pnl", 0)
        
        # Ajouter à la courbe d'équité
        self.equity_curve.loc[timestamp] = portfolio_value
    
    def get_position_value(self):
        """
        Calcule la valeur totale des positions ouvertes
        
        Returns:
            float: Valeur totale des positions
        """
        position_value = 0
        for symbol, position in self.positions.items():
            position_value += position["size"] * position["current_price"]
        return position_value
    
    def get_portfolio_value(self):
        """
        Calcule la valeur totale du portefeuille (capital + positions)
        
        Returns:
            float: Valeur totale du portefeuille
        """
        return self.current_capital + self.get_position_value()
    
    def get_position_exposure(self):
        """
        Calcule l'exposition des positions par rapport au capital total
        
        Returns:
            float: Exposition en pourcentage
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value == 0:
            return 0
        return self.get_position_value() / portfolio_value
    
    def get_available_capital(self):
        """
        Calcule le capital disponible pour de nouvelles positions
        
        Returns:
            float: Capital disponible
        """
        return self.current_capital
    
    def get_max_position_size(self, symbol, price):
        """
        Calcule la taille maximale pour une nouvelle position
        
        Args:
            symbol (str): Symbole de la paire
            price (float): Prix actuel
            
        Returns:
            float: Taille maximale de position
        """
        # Calcul basique basé sur le capital disponible et le prix
        available_capital = self.get_available_capital()
        max_position_value = available_capital * 0.2  # Limite à 20% du capital disponible
        max_size = max_position_value / price
        
        return max_size
    
    def get_performance_stats(self):
        """
        Calcule les statistiques de performance du portefeuille
        
        Returns:
            dict: Statistiques de performance
        """
        if len(self.transactions) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "return_pct": 0
            }
        
        # Calculer les statistiques sur les transactions fermées
        closed_trades = [t for t in self.transactions if t["type"] == "close"]
        
        # Nombre total de trades
        total_trades = len(closed_trades)
        
        # Trades gagnants/perdants
        winning_trades = [t for t in closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in closed_trades if t["pnl"] <= 0]
        
        # Taux de réussite
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit Factor
        total_profit = sum([t["pnl"] for t in winning_trades])
        total_loss = abs(sum([t["pnl"] for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Moyenne des gains/pertes
        average_win = total_profit / len(winning_trades) if len(winning_trades) > 0 else 0
        average_loss = total_loss / len(losing_trades) if len(losing_trades) > 0 else 0
        
        # Plus grand gain/perte
        largest_win = max([t["pnl"] for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t["pnl"] for t in losing_trades]) if losing_trades else 0
        
        # Maximum Drawdown
        equity = self.equity_curve['equity']
        peak = equity.cummax()
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # Rendements quotidiens
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        
        # Ratio de Sharpe (rendement annualisé / écart-type annualisé)
        sharpe_ratio = 0
        if len(daily_returns) > 0:
            annualized_return = daily_returns.mean() * 252
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Rendement total en pourcentage
        return_pct = (self.current_capital / self.initial_capital - 1) * 100
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "return_pct": return_pct
        }
    
    def get_positions_summary(self):
        """
        Résume les positions actuelles
        
        Returns:
            list: Résumé des positions
        """
        positions_summary = []
        
        for symbol, position in self.positions.items():
            summary = {
                "symbol": symbol,
                "side": position["side"],
                "size": position["size"],
                "entry_price": position["entry_price"],
                "current_price": position["current_price"],
                "unrealized_pnl": position["unrealized_pnl"],
                "unrealized_pnl_pct": position["unrealized_pnl_pct"]
            }
            positions_summary.append(summary)
            
        return positions_summary
    
    def get_equity_curve(self):
        """
        Retourne la courbe d'équité
        
        Returns:
            pd.DataFrame: Courbe d'équité
        """
        return self.equity_curve
    
    def reset(self):
        """
        Réinitialise le portefeuille
        """
        self.current_capital = self.initial_capital
        self.positions = {}
        self.transactions = []
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])
        self.equity_curve.loc[0] = [datetime.now(), self.initial_capital]
        self.equity_curve.set_index('timestamp', inplace=True)
        
        logger.info("Portefeuille réinitialisé")