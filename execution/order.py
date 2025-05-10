"""
Module de gestion des ordres
Définit les différents types d'ordres (market, limit, stop loss, etc.)
"""
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Order:
    """
    Classe représentant un ordre de trading
    """
    
    def __init__(self, symbol, side, order_type, amount, price=None, stop_price=None, order_id=None):
        """
        Initialise un ordre
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            side (str): Côté de l'ordre ('buy' ou 'sell')
            order_type (str): Type d'ordre ('market', 'limit', 'stop_loss', 'take_profit')
            amount (float): Quantité à acheter/vendre
            price (float, optional): Prix limite
            stop_price (float, optional): Prix d'activation pour stop loss/take profit
            order_id (str, optional): ID de l'ordre
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id or f"order-{int(time.time() * 1000)}"
        
        self.status = "created"
        self.filled_amount = 0
        self.remaining_amount = amount
        self.avg_fill_price = None
        self.executed_time = None
        self.create_time = datetime.now()
        
        logger.info(f"Ordre créé: {self.order_type} {self.side} {self.amount} {self.symbol}" +
                   (f" @ {self.price}" if self.price else "") +
                   (f" (stop: {self.stop_price})" if self.stop_price else ""))
    
    def update_status(self, status, filled_amount=None, avg_fill_price=None):
        """
        Met à jour le statut de l'ordre
        
        Args:
            status (str): Nouveau statut ('open', 'filled', 'partially_filled', 'canceled', 'rejected')
            filled_amount (float, optional): Quantité remplie
            avg_fill_price (float, optional): Prix moyen de remplissage
        """
        if status == self.status:
            return
            
        old_status = self.status
        self.status = status
        
        if filled_amount is not None:
            self.filled_amount = filled_amount
            self.remaining_amount = self.amount - filled_amount
            
        if avg_fill_price is not None:
            self.avg_fill_price = avg_fill_price
            
        if status in ['filled', 'canceled', 'rejected']:
            self.executed_time = datetime.now()
            
        logger.info(f"Statut de l'ordre {self.order_id} mis à jour: {old_status} -> {status}" +
                   (f", rempli: {filled_amount}/{self.amount}" if filled_amount is not None else "") +
                   (f", prix moyen: {avg_fill_price}" if avg_fill_price is not None else ""))
    
    def is_active(self):
        """
        Vérifie si l'ordre est actif
        
        Returns:
            bool: True si l'ordre est actif, False sinon
        """
        return self.status in ['created', 'open', 'partially_filled']
    
    def is_filled(self):
        """
        Vérifie si l'ordre est entièrement rempli
        
        Returns:
            bool: True si l'ordre est entièrement rempli, False sinon
        """
        return self.status == 'filled'
    
    def get_execution_time(self):
        """
        Calcule le temps d'exécution de l'ordre
        
        Returns:
            float: Temps d'exécution en secondes
        """
        if self.executed_time is None:
            return None
            
        return (self.executed_time - self.create_time).total_seconds()
    
    def to_dict(self):
        """
        Convertit l'ordre en dictionnaire
        
        Returns:
            dict: Ordre sous forme de dictionnaire
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'amount': self.amount,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status,
            'filled_amount': self.filled_amount,
            'remaining_amount': self.remaining_amount,
            'avg_fill_price': self.avg_fill_price,
            'create_time': self.create_time.isoformat(),
            'executed_time': self.executed_time.isoformat() if self.executed_time else None
        }
    
    def __str__(self):
        """
        Représentation textuelle de l'ordre
        
        Returns:
            str: Description de l'ordre
        """
        return (f"Ordre {self.order_id}: {self.order_type} {self.side} {self.amount} {self.symbol}" +
               (f" @ {self.price}" if self.price else "") +
               (f" (stop: {self.stop_price})" if self.stop_price else "") +
               f" - {self.status}")


# Fonctions de création d'ordres
def create_market_order(symbol, side, amount):
    """
    Crée un ordre au marché
    
    Args:
        symbol (str): Symbole de la paire (ex: BTC/USDT)
        side (str): Côté de l'ordre ('buy' ou 'sell')
        amount (float): Quantité à acheter/vendre
        
    Returns:
        Order: Ordre créé
    """
    return Order(symbol, side, 'market', amount)

def create_limit_order(symbol, side, amount, price):
    """
    Crée un ordre à cours limité
    
    Args:
        symbol (str): Symbole de la paire (ex: BTC/USDT)
        side (str): Côté de l'ordre ('buy' ou 'sell')
        amount (float): Quantité à acheter/vendre
        price (float): Prix limite
        
    Returns:
        Order: Ordre créé
    """
    return Order(symbol, side, 'limit', amount, price=price)

def create_stop_loss_order(symbol, side, amount, stop_price, price=None):
    """
    Crée un ordre stop loss
    
    Args:
        symbol (str): Symbole de la paire (ex: BTC/USDT)
        side (str): Côté de l'ordre ('buy' ou 'sell')
        amount (float): Quantité à acheter/vendre
        stop_price (float): Prix d'activation
        price (float, optional): Prix limite (pour stop limit)
        
    Returns:
        Order: Ordre créé
    """
    return Order(symbol, side, 'stop_loss', amount, price=price, stop_price=stop_price)

def create_take_profit_order(symbol, side, amount, stop_price, price=None):
    """
    Crée un ordre take profit
    
    Args:
        symbol (str): Symbole de la paire (ex: BTC/USDT)
        side (str): Côté de l'ordre ('buy' ou 'sell')
        amount (float): Quantité à acheter/vendre
        stop_price (float): Prix d'activation
        price (float, optional): Prix limite (pour take profit limit)
        
    Returns:
        Order: Ordre créé
    """
    return Order(symbol, side, 'take_profit', amount, price=price, stop_price=stop_price)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer différents types d'ordres
    market_order = create_market_order("BTC/USDT", "buy", 0.1)
    limit_order = create_limit_order("ETH/USDT", "sell", 1.5, 3500)
    stop_loss = create_stop_loss_order("BTC/USDT", "sell", 0.1, 55000)
    take_profit = create_take_profit_order("ETH/USDT", "sell", 1.5, 4000)
    
    # Mettre à jour le statut des ordres
    market_order.update_status("open")
    market_order.update_status("filled", 0.1, 60000)
    
    limit_order.update_status("open")
    limit_order.update_status("partially_filled", 0.5, 3500)
    limit_order.update_status("filled", 1.5, 3500)
    
    stop_loss.update_status("open")
    stop_loss.update_status("canceled")
    
    # Afficher les informations sur les ordres
    print(f"Ordre au marché: {market_order}")
    print(f"Temps d'exécution: {market_order.get_execution_time()} secondes")
    
    print(f"\nOrdre à cours limité: {limit_order}")
    print(f"Temps d'exécution: {limit_order.get_execution_time()} secondes")
    
    print(f"\nOrdre stop loss: {stop_loss}")
    print(f"Temps d'exécution: {stop_loss.get_execution_time()} secondes")
    
    print(f"\nOrdre take profit: {take_profit}")
    print(f"Est actif: {take_profit.is_active()}")