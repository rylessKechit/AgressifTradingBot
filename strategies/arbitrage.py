"""
Stratégie d'arbitrage
Implémente une stratégie d'arbitrage exploitant les différences de prix
entre plusieurs marchés ou instruments financiers
"""
import pandas as pd
import numpy as np
import logging
import ccxt
from strategies.base_strategy import BaseStrategy
from config.strategy_params import ARBITRAGE_PARAMS

logger = logging.getLogger(__name__)

class ArbitrageStrategy(BaseStrategy):
    """
    Stratégie qui exploite les différences de prix entre différents marchés
    ou instruments financiers pour générer des profits sans risque
    """
    
    def __init__(self, params=None):
        """
        Initialise la stratégie d'arbitrage
        
        Args:
            params (dict, optional): Paramètres personnalisés pour la stratégie
        """
        # Charger les paramètres par défaut
        default_params = ARBITRAGE_PARAMS.copy()
        
        # Fusionner avec les paramètres personnalisés s'ils existent
        if params:
            default_params.update(params)
        
        super().__init__("Arbitrage", default_params)
        
        # Initialiser les exchanges pour l'arbitrage
        self.exchanges = {}
        self.initialize_exchanges()
        
        logger.info(f"Stratégie d'arbitrage initialisée avec {len(self.exchanges)} exchanges")
    
    def initialize_exchanges(self):
        """
        Initialise les connexions aux exchanges pour l'arbitrage
        """
        exchanges_list = self.params.get('exchanges', [])
        
        for exchange_id in exchanges_list:
            try:
                # Créer l'instance de l'exchange
                exchange = getattr(ccxt, exchange_id)({
                    'enableRateLimit': True,
                })
                
                self.exchanges[exchange_id] = exchange
                logger.info(f"Exchange {exchange_id} initialisé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'exchange {exchange_id}: {e}")
    
    def preprocess_data(self, data):
        """
        Prétraite les données pour la stratégie
        
        Args:
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.DataFrame: DataFrame prétraité
        """
        # Pas de prétraitement spécifique nécessaire pour cette stratégie
        return data
    
    def generate_signals(self, data):
        """
        Génère les signaux de trading basés sur l'arbitrage
        
        Args:
            data (pd.DataFrame): DataFrame avec les données et indicateurs
            
        Returns:
            pd.Series: Série contenant les signaux (1=achat, -1=vente, 0=neutre)
        """
        # Initialiser les signaux à zéro
        signals = pd.Series(0, index=data.index)
        
        # Pour un backtest réel, nous aurions besoin de données de plusieurs exchanges
        # Dans cette implémentation, nous allons simuler des opportunités d'arbitrage
        # basées sur les données historiques d'un seul exchange
        
        # Ajouter un peu de bruit aléatoire au prix pour simuler des différences de prix
        # entre différents exchanges
        np.random.seed(42)  # Pour la reproductibilité
        noise_factor = 0.001  # 0.1% de bruit
        
        # Simuler les prix sur différents exchanges
        exchange1_prices = data['close'].copy()
        exchange2_prices = data['close'] * (1 + np.random.normal(0, noise_factor, len(data)))
        exchange3_prices = data['close'] * (1 + np.random.normal(0, noise_factor, len(data)))
        
        # Calculer les écarts de prix
        spread_1_2 = (exchange2_prices - exchange1_prices) / exchange1_prices * 100
        spread_1_3 = (exchange3_prices - exchange1_prices) / exchange1_prices * 100
        spread_2_3 = (exchange3_prices - exchange2_prices) / exchange2_prices * 100
        
        # Seuil minimum pour l'arbitrage (en pourcentage)
        min_spread = self.params.get('min_spread_pct', 0.5)
        
        # Frais estimés par exchange (en pourcentage)
        exchange_fees = self.params.get('exchange_fees', {})
        fee1 = exchange_fees.get('exchange1', 0.1) / 100
        fee2 = exchange_fees.get('exchange2', 0.1) / 100
        fee3 = exchange_fees.get('exchange3', 0.1) / 100
        
        # Calculer le coût total des frais pour une transaction d'arbitrage
        total_fees = fee1 + fee2 + fee3
        
        # Identifier les opportunités d'arbitrage
        for i in range(len(data)):
            # Vérifier s'il y a une opportunité d'arbitrage entre exchange1 et exchange2
            if spread_1_2.iloc[i] > min_spread + total_fees:
                # Acheter sur exchange1, vendre sur exchange2
                signals.iloc[i] = 1
            elif spread_1_2.iloc[i] < -(min_spread + total_fees):
                # Acheter sur exchange2, vendre sur exchange1
                signals.iloc[i] = -1
                
            # Vérifier s'il y a une opportunité d'arbitrage entre exchange1 et exchange3
            elif spread_1_3.iloc[i] > min_spread + total_fees:
                # Acheter sur exchange1, vendre sur exchange3
                signals.iloc[i] = 1
            elif spread_1_3.iloc[i] < -(min_spread + total_fees):
                # Acheter sur exchange3, vendre sur exchange1
                signals.iloc[i] = -1
                
            # Vérifier s'il y a une opportunité d'arbitrage entre exchange2 et exchange3
            elif spread_2_3.iloc[i] > min_spread + total_fees:
                # Acheter sur exchange2, vendre sur exchange3
                signals.iloc[i] = 1
            elif spread_2_3.iloc[i] < -(min_spread + total_fees):
                # Acheter sur exchange3, vendre sur exchange2
                signals.iloc[i] = -1
        
        return signals
    
    def postprocess_signals(self, signals, data):
        """
        Post-traite les signaux générés pour éliminer les faux signaux
        
        Args:
            signals (pd.Series): Signaux générés
            data (pd.DataFrame): DataFrame avec les données
            
        Returns:
            pd.Series: Signaux post-traités
        """
        # Pour l'arbitrage, nous ne voulons pas de signaux consécutifs
        # car chaque opportunité d'arbitrage devrait être indépendante
        return self.apply_cooldown(signals, cooldown_period=self.params.get('cooldown_period', 3))
    
    def get_arbitrage_opportunities(self, symbol):
        """
        Recherche des opportunités d'arbitrage en temps réel pour un symbole donné
        
        Args:
            symbol (str): Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            list: Liste des opportunités d'arbitrage
        """
        if not self.exchanges:
            logger.warning("Aucun exchange initialisé pour l'arbitrage")
            return []
            
        # Récupérer les tickers pour le symbole sur tous les exchanges
        tickers = {}
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                tickers[exchange_id] = ticker
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du ticker pour {symbol} sur {exchange_id}: {e}")
        
        if len(tickers) < 2:
            logger.warning(f"Pas assez de tickers pour {symbol}, impossible de rechercher des opportunités d'arbitrage")
            return []
            
        # Rechercher des opportunités d'arbitrage
        opportunities = []
        
        # Frais estimés par exchange (en pourcentage)
        exchange_fees = self.params.get('exchange_fees', {})
        
        # Seuil minimum pour l'arbitrage (en pourcentage)
        min_spread = self.params.get('min_spread_pct', 0.5)
        
        # Comparer les prix sur tous les exchanges
        exchange_ids = list(tickers.keys())
        
        for i in range(len(exchange_ids)):
            for j in range(i + 1, len(exchange_ids)):
                exchange1 = exchange_ids[i]
                exchange2 = exchange_ids[j]
                
                bid1 = tickers[exchange1]['bid']
                ask1 = tickers[exchange1]['ask']
                bid2 = tickers[exchange2]['bid']
                ask2 = tickers[exchange2]['ask']
                
                # Frais pour ces exchanges
                fee1 = exchange_fees.get(exchange1, 0.1) / 100
                fee2 = exchange_fees.get(exchange2, 0.1) / 100
                
                # Coût total des frais
                total_fees = fee1 + fee2
                
                # Calculer les écarts de prix
                # Acheter sur exchange1, vendre sur exchange2
                spread1 = (bid2 * (1 - fee2)) - (ask1 * (1 + fee1))
                spread1_pct = spread1 / ask1 * 100
                
                # Acheter sur exchange2, vendre sur exchange1
                spread2 = (bid1 * (1 - fee1)) - (ask2 * (1 + fee2))
                spread2_pct = spread2 / ask2 * 100
                
                # Vérifier si les écarts dépassent le seuil minimum
                if spread1_pct > min_spread:
                    # Opportunité d'arbitrage: acheter sur exchange1, vendre sur exchange2
                    opportunities.append({
                        'type': 'arbitrage',
                        'buy_exchange': exchange1,
                        'sell_exchange': exchange2,
                        'buy_price': ask1,
                        'sell_price': bid2,
                        'spread': spread1,
                        'spread_pct': spread1_pct,
                        'fees': total_fees * 100,  # en pourcentage
                        'net_profit_pct': spread1_pct - (total_fees * 100)
                    })
                    
                if spread2_pct > min_spread:
                    # Opportunité d'arbitrage: acheter sur exchange2, vendre sur exchange1
                    opportunities.append({
                        'type': 'arbitrage',
                        'buy_exchange': exchange2,
                        'sell_exchange': exchange1,
                        'buy_price': ask2,
                        'sell_price': bid1,
                        'spread': spread2,
                        'spread_pct': spread2_pct,
                        'fees': total_fees * 100,  # en pourcentage
                        'net_profit_pct': spread2_pct - (total_fees * 100)
                    })
        
        # Trier les opportunités par profit net décroissant
        opportunities.sort(key=lambda x: x['net_profit_pct'], reverse=True)
        
        return opportunities
    
    def triangular_arbitrage(self, base_currency='USDT'):
        """
        Recherche des opportunités d'arbitrage triangulaire
        
        Args:
            base_currency (str): Devise de base (ex: USDT, BTC)
            
        Returns:
            list: Liste des opportunités d'arbitrage triangulaire
        """
        if not self.exchanges:
            logger.warning("Aucun exchange initialisé pour l'arbitrage triangulaire")
            return []
            
        # Utiliser le premier exchange disponible
        exchange_id = list(self.exchanges.keys())[0]
        exchange = self.exchanges[exchange_id]
        
        try:
            # Récupérer les marchés
            markets = exchange.fetch_markets()
            
            # Filtrer les marchés actifs
            active_markets = [market for market in markets if market['active']]
            
            # Créer un dictionnaire des paires
            pairs = {}
            for market in active_markets:
                symbol = market['symbol']
                base = market['base']
                quote = market['quote']
                
                if base not in pairs:
                    pairs[base] = []
                pairs[base].append({'symbol': symbol, 'quote': quote})
                
                if quote not in pairs:
                    pairs[quote] = []
                pairs[quote].append({'symbol': symbol, 'base': base})
            
            # Récupérer les tickers
            tickers = exchange.fetch_tickers()
            
            # Rechercher des opportunités d'arbitrage triangulaire
            opportunities = []
            
            # Frais par transaction (en pourcentage)
            fee = exchange.markets[active_markets[0]['symbol']].get('taker', 0.1) / 100
            
            # Pour chaque paire avec la devise de base
            for pair1 in pairs.get(base_currency, []):
                quote1 = pair1['quote']
                
                # Pour chaque paire avec la devise intermédiaire
                for pair2 in pairs.get(quote1, []):
                    if 'base' in pair2 and pair2['base'] == quote1:
                        quote2 = pair2['quote']
                        
                        # Vérifier s'il existe une paire pour fermer le triangle
                        for pair3 in pairs.get(quote2, []):
                            if 'base' in pair3 and pair3['base'] == quote2 and 'quote' in pair3 and pair3['quote'] == base_currency:
                                # Triangle trouvé: base_currency -> quote1 -> quote2 -> base_currency
                                
                                # Récupérer les prix
                                symbol1 = pair1['symbol']
                                symbol2 = pair2['symbol']
                                symbol3 = pair3['symbol']
                                
                                if symbol1 in tickers and symbol2 in tickers and symbol3 in tickers:
                                    # Prix pour le premier trade (base_currency -> quote1)
                                    price1 = tickers[symbol1]['ask']
                                    
                                    # Prix pour le deuxième trade (quote1 -> quote2)
                                    price2 = tickers[symbol2]['ask']
                                    
                                    # Prix pour le troisième trade (quote2 -> base_currency)
                                    price3 = tickers[symbol3]['bid']
                                    
                                    # Calcul du profit potentiel
                                    # Commencer avec 1 unité de base_currency
                                    amount1 = 1.0
                                    
                                    # Premier trade: base_currency -> quote1
                                    amount2 = amount1 / price1 * (1 - fee)
                                    
                                    # Deuxième trade: quote1 -> quote2
                                    amount3 = amount2 / price2 * (1 - fee)
                                    
                                    # Troisième trade: quote2 -> base_currency
                                    amount4 = amount3 * price3 * (1 - fee)
                                    
                                    # Profit ou perte
                                    profit = amount4 - amount1
                                    profit_pct = (profit / amount1) * 100
                                    
                                    # Vérifier s'il y a un profit après les frais
                                    if profit_pct > self.params.get('min_spread_pct', 0.5):
                                        opportunities.append({
                                            'type': 'triangular',
                                            'exchange': exchange_id,
                                            'path': f"{base_currency} -> {quote1} -> {quote2} -> {base_currency}",
                                            'trades': [
                                                {'symbol': symbol1, 'action': 'buy', 'price': price1},
                                                {'symbol': symbol2, 'action': 'buy', 'price': price2},
                                                {'symbol': symbol3, 'action': 'sell', 'price': price3}
                                            ],
                                            'profit': profit,
                                            'profit_pct': profit_pct,
                                            'fees': fee * 3 * 100  # Frais totaux en pourcentage
                                        })
            
            # Trier les opportunités par profit décroissant
            opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'opportunités d'arbitrage triangulaire: {e}")
            return []


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialiser la stratégie
    strategy = ArbitrageStrategy()
    
    # Simuler quelques données historiques
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Créer des données simulées
    dates = pd.date_range(start='2022-01-01', end='2022-01-10', freq='1h')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.cumsum(np.random.normal(0, 100, len(dates)))
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.normal(0, 50, len(dates)),
        'low': prices - np.random.normal(0, 50, len(dates)),
        'close': prices + np.random.normal(0, 20, len(dates)),
        'volume': np.random.normal(100, 20, len(dates))
    }, index=dates)
    
    # Générer les signaux
    signals = strategy.run(data)
    
    # Afficher les résultats
    print(f"Nombre total de signaux: {len(signals[signals != 0])}")
    print(f"Signaux d'achat: {len(signals[signals == 1])}")
    print(f"Signaux de vente: {len(signals[signals == -1])}")
    
    # Rechercher des opportunités d'arbitrage en temps réel
    if strategy.exchanges:
        opportunities = strategy.get_arbitrage_opportunities("BTC/USDT")
        print(f"\nOpportunités d'arbitrage trouvées: {len(opportunities)}")
        
        for i, opp in enumerate(opportunities[:3]):  # Afficher les 3 meilleures opportunités
            print(f"\nOpportunité #{i+1}:")
            print(f"Acheter sur {opp['buy_exchange']} à {opp['buy_price']}")
            print(f"Vendre sur {opp['sell_exchange']} à {opp['sell_price']}")
            print(f"Spread: {opp['spread_pct']:.2f}%, Frais: {opp['fees']:.2f}%, Profit net: {opp['net_profit_pct']:.2f}%")
            
        # Rechercher des opportunités d'arbitrage triangulaire
        triangular_opps = strategy.triangular_arbitrage()
        print(f"\nOpportunités d'arbitrage triangulaire trouvées: {len(triangular_opps)}")
        
        for i, opp in enumerate(triangular_opps[:3]):  # Afficher les 3 meilleures opportunités
            print(f"\nOpportunité triangulaire #{i+1}:")
            print(f"Chemin: {opp['path']}")
            print(f"Profit: {opp['profit_pct']:.2f}%, Frais: {opp['fees']:.2f}%")