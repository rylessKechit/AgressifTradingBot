"""
Module pour le calcul des indicateurs techniques
Fournit des fonctions pour calculer divers indicateurs techniques sur les données de marché
"""
import pandas as pd
import numpy as np
import logging
import talib
from scipy import stats

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Classe pour calculer les indicateurs techniques sur un DataFrame
    """
    
    @staticmethod
    def add_all_indicators(df, include_patterns=True):
        """
        Ajoute tous les indicateurs techniques au DataFrame
        """
        if df.empty:
            logger.warning("DataFrame vide, impossible d'ajouter les indicateurs")
            return df
            
        # Vérifier les colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Colonne {col} manquante dans le DataFrame")
                return df
                
        # Copie du DataFrame pour éviter les warnings de fragmentation
        df_copy = df.copy()
        
        # Convertir en numpy arrays pour talib
        open_price = df_copy['open'].values
        high_price = df_copy['high'].values
        low_price = df_copy['low'].values
        close_price = df_copy['close'].values
        volume = df_copy['volume'].values
        
        try:
            # Créer des dictionnaires pour stocker les valeurs des indicateurs
            trend_indicators = {}
            momentum_indicators = {}
            volatility_indicators = {}
            volume_indicators = {}
            
            # --- Moyennes Mobiles (important pour les stratégies de tendance) ---
            # EMA (Exponential Moving Average)
            for period in [5, 8, 10, 12, 15, 20, 21, 25, 30, 50, 100, 200]:
                trend_indicators[f'ema_{period}'] = talib.EMA(close_price, timeperiod=period)
            
            # SMA (Simple Moving Average)
            for period in [5, 8, 10, 20, 21, 50, 100, 200]:
                trend_indicators[f'sma_{period}'] = talib.SMA(close_price, timeperiod=period)
            
            # Crosses
            trend_indicators['ema_8_21_cross'] = np.where(
                trend_indicators['ema_8'] > trend_indicators['ema_21'], 1, -1)
            trend_indicators['ema_50_200_cross'] = np.where(
                trend_indicators['ema_50'] > trend_indicators['ema_200'], 1, -1)
            
            # --- MACD ---
            macd, macd_signal, macd_hist = talib.MACD(
                close_price, fastperiod=12, slowperiod=26, signalperiod=9)
            trend_indicators['macd'] = macd
            trend_indicators['macd_signal'] = macd_signal
            trend_indicators['macd_hist'] = macd_hist
            
            # --- RSI ---
            for period in [7, 14, 21]:
                momentum_indicators[f'rsi_{period}'] = talib.RSI(close_price, timeperiod=period)
            
            # --- Bollinger Bands ---
            upper, middle, lower = talib.BBANDS(
                close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            volatility_indicators['bb_upper'] = upper
            volatility_indicators['bb_middle'] = middle
            volatility_indicators['bb_lower'] = lower
            
            # --- ATR (Average True Range) ---
            for period in [7, 14, 21]:
                volatility_indicators[f'atr_{period}'] = talib.ATR(
                    high_price, low_price, close_price, timeperiod=period)
            
            # --- ADX (Average Directional Index) ---
            trend_indicators['adx'] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
            trend_indicators['adx_plus_di'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
            trend_indicators['adx_minus_di'] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
            
            # --- Stochastic ---
            momentum_indicators['stoch_k'], momentum_indicators['stoch_d'] = talib.STOCH(
                high_price, low_price, close_price, 
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            
            # --- Volumes ---
            volume_indicators['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            volume_indicators['volume_ratio'] = volume / volume_indicators['volume_sma_20']
            
            # Ajouter tous les indicateurs au DataFrame en une seule fois
            # (évite la fragmentation)
            all_indicators = {}
            all_indicators.update(trend_indicators)
            all_indicators.update(momentum_indicators)
            all_indicators.update(volatility_indicators)
            all_indicators.update(volume_indicators)
            
            # Créer un DataFrame avec tous les indicateurs
            indicators_df = pd.DataFrame(all_indicators, index=df_copy.index)
            
            # Combiner avec le DataFrame original
            df_with_indicators = pd.concat([df_copy, indicators_df], axis=1)
            
            # Ajout des indicateurs supplémentaires qui nécessitent les autres indicateurs
            if include_patterns:
                pattern_signals = {}
                # Patterns de chandeliers
                for pattern_function in [
                    talib.CDL2CROWS, talib.CDL3BLACKCROWS, talib.CDL3WHITESOLDIERS,
                    talib.CDLENGULFING, talib.CDLHAMMER, talib.CDLMORNINGSTAR
                ]:
                    pattern_name = pattern_function.__name__[3:].lower()  # Enlève 'CDL' du nom
                    pattern_signals[pattern_name] = pattern_function(
                        open_price, high_price, low_price, close_price)
                
                # Ajouter les patterns au DataFrame
                patterns_df = pd.DataFrame(pattern_signals, index=df_copy.index)
                df_with_indicators = pd.concat([df_with_indicators, patterns_df], axis=1)
            
            # Supprimer les NaN
            df_with_indicators.fillna(method='ffill', inplace=True)
            df_with_indicators.fillna(method='bfill', inplace=True)
            
            # Vérification finale
            logger.info(f"Indicateurs ajoutés: {df_with_indicators.shape[1] - df_copy.shape[1]} colonnes")
            
            return df_with_indicators
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des indicateurs: {e}")
            return df_copy

    @staticmethod
    def add_moving_averages(df):
        """
        Ajoute les moyennes mobiles au DataFrame
        """
        close_price = df['close'].values
        
        # SMA (Simple Moving Average)
        df['sma_5'] = talib.SMA(close_price, timeperiod=5)
        df['sma_8'] = talib.SMA(close_price, timeperiod=8)  # Ajouté
        df['sma_10'] = talib.SMA(close_price, timeperiod=10)
        df['sma_20'] = talib.SMA(close_price, timeperiod=20)
        df['sma_21'] = talib.SMA(close_price, timeperiod=21)  # Ajouté
        df['sma_50'] = talib.SMA(close_price, timeperiod=50)
        df['sma_100'] = talib.SMA(close_price, timeperiod=100)
        df['sma_200'] = talib.SMA(close_price, timeperiod=200)
        
        # EMA (Exponential Moving Average)
        df['ema_5'] = talib.EMA(close_price, timeperiod=5)
        df['ema_8'] = talib.EMA(close_price, timeperiod=8)  # Ajouté
        df['ema_10'] = talib.EMA(close_price, timeperiod=10)
        df['ema_20'] = talib.EMA(close_price, timeperiod=20)
        df['ema_21'] = talib.EMA(close_price, timeperiod=21)  # Ajouté
        df['ema_50'] = talib.EMA(close_price, timeperiod=50)
        df['ema_100'] = talib.EMA(close_price, timeperiod=100)
        df['ema_200'] = talib.EMA(close_price, timeperiod=200)
        
        # WMA (Weighted Moving Average)
        df['wma_20'] = talib.WMA(close_price, timeperiod=20)
        
        # TEMA (Triple Exponential Moving Average)
        df['tema_20'] = talib.TEMA(close_price, timeperiod=20)
        
        # Crosses and slopes
        df['ema_5_10_cross'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)
        df['ema_8_21_cross'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)  # Ajouté
        df['ema_10_20_cross'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)
        df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
        
        # Calcul des pentes (slopes)
        for ma in ['ema_8', 'ema_10', 'ema_20', 'ema_21', 'ema_50', 'ema_200']:
            df[f'{ma}_slope'] = talib.LINEARREG_SLOPE(df[ma].values, timeperiod=5)
        
        return df
    
    @staticmethod
    def add_macd(df):
        """
        Ajoute l'indicateur MACD au DataFrame
        """
        close_price = df['close'].values
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(
            close_price, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_cross'] = np.where(macd > macd_signal, 1, -1)
        
        return df
    
    @staticmethod
    def add_adx(df):
        """
        Ajoute l'indicateur ADX (Average Directional Index) au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
        df['adx_plus_di'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
        df['adx_minus_di'] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
        
        # Cross du +DI et -DI
        df['adx_di_cross'] = np.where(df['adx_plus_di'] > df['adx_minus_di'], 1, -1)
        
        # Trend strength
        df['adx_trend_strength'] = np.select(
            [df['adx'] < 20, (df['adx'] >= 20) & (df['adx'] < 40), df['adx'] >= 40],
            ['weak', 'moderate', 'strong'],
            default='weak'
        )
        
        return df
    
    @staticmethod
    def add_ichimoku(df):
        """
        Ajoute l'indicateur Ichimoku Kinko Hyo au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        
        # Conversion Line (Tenkan-sen)
        high_9 = pd.Series(high_price).rolling(window=9).max()
        low_9 = pd.Series(low_price).rolling(window=9).min()
        df['ichimoku_tenkan_sen'] = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = pd.Series(high_price).rolling(window=26).max()
        low_26 = pd.Series(low_price).rolling(window=26).min()
        df['ichimoku_kijun_sen'] = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        high_52 = pd.Series(high_price).rolling(window=52).max()
        low_52 = pd.Series(low_price).rolling(window=52).min()
        df['ichimoku_senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Lagging Span (Chikou Span)
        df['ichimoku_chikou_span'] = pd.Series(df['close'].values).shift(-26)
        
        # Cloud (Kumo) signal
        df['ichimoku_cloud_green'] = np.where(df['ichimoku_senkou_span_a'] > df['ichimoku_senkou_span_b'], 1, 0)
        df['ichimoku_cloud_red'] = np.where(df['ichimoku_senkou_span_a'] < df['ichimoku_senkou_span_b'], 1, 0)
        
        # Price above/below cloud
        df['ichimoku_price_above_cloud'] = np.where(
            (df['close'] > df['ichimoku_senkou_span_a']) & (df['close'] > df['ichimoku_senkou_span_b']), 
            1, 0
        )
        df['ichimoku_price_below_cloud'] = np.where(
            (df['close'] < df['ichimoku_senkou_span_a']) & (df['close'] < df['ichimoku_senkou_span_b']), 
            1, 0
        )
        
        # TK Cross
        df['ichimoku_tk_cross'] = np.where(df['ichimoku_tenkan_sen'] > df['ichimoku_kijun_sen'], 1, -1)
        
        return df
    
    @staticmethod
    def add_rsi(df):
        """
        Ajoute l'indicateur RSI (Relative Strength Index) au DataFrame
        """
        close_price = df['close'].values
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = talib.RSI(close_price, timeperiod=14)
        df['rsi_7'] = talib.RSI(close_price, timeperiod=7)
        df['rsi_21'] = talib.RSI(close_price, timeperiod=21)
        
        # RSI zones
        df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
        df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
        
        # RSI divergence (price up, RSI down or vice versa)
        df['close_higher'] = df['close'] > df['close'].shift(1)
        df['rsi_lower'] = df['rsi_14'] < df['rsi_14'].shift(1)
        df['rsi_bearish_div'] = np.where((df['close_higher']) & (df['rsi_lower']), 1, 0)
        
        df['close_lower'] = df['close'] < df['close'].shift(1)
        df['rsi_higher'] = df['rsi_14'] > df['rsi_14'].shift(1)
        df['rsi_bullish_div'] = np.where((df['close_lower']) & (df['rsi_higher']), 1, 0)
        
        return df
    
    @staticmethod
    def add_stochastic(df):
        """
        Ajoute l'indicateur Stochastique au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            high_price, low_price, close_price, 
            fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        
        # Stochastic Zones
        df['stoch_overbought'] = np.where(df['stoch_k'] > 80, 1, 0)
        df['stoch_oversold'] = np.where(df['stoch_k'] < 20, 1, 0)
        
        # Stochastic Cross
        df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
        
        return df
    
    @staticmethod
    def add_cci(df):
        """
        Ajoute l'indicateur CCI (Commodity Channel Index) au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # CCI (Commodity Channel Index)
        df['cci_14'] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
        df['cci_20'] = talib.CCI(high_price, low_price, close_price, timeperiod=20)
        
        # CCI Zones
        df['cci_overbought'] = np.where(df['cci_14'] > 100, 1, 0)
        df['cci_oversold'] = np.where(df['cci_14'] < -100, 1, 0)
        
        return df
    
    @staticmethod
    def add_williams_r(df):
        """
        Ajoute l'indicateur Williams %R au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # Williams %R
        df['willr_14'] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
        
        # Williams %R Zones
        df['willr_overbought'] = np.where(df['willr_14'] > -20, 1, 0)
        df['willr_oversold'] = np.where(df['willr_14'] < -80, 1, 0)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df):
        """
        Ajoute les Bandes de Bollinger au DataFrame
        """
        close_price = df['close'].values
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            close_price, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        # Bollinger Bands Width & %B
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Bands Signals
        df['bb_above_upper'] = np.where(df['close'] > df['bb_upper'], 1, 0)
        df['bb_below_lower'] = np.where(df['close'] < df['bb_lower'], 1, 0)
        df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(window=50).mean() * 0.8, 1, 0)
        
        return df
    
    @staticmethod
    def add_atr(df):
        """
        Ajoute l'indicateur ATR (Average True Range) au DataFrame
        """
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(high_price, low_price, close_price, timeperiod=14)
        df['atr_7'] = talib.ATR(high_price, low_price, close_price, timeperiod=7)
        df['atr_21'] = talib.ATR(high_price, low_price, close_price, timeperiod=21)
        
        # ATR en pourcentage du prix
        df['atr_pct_14'] = df['atr_14'] / df['close'] * 100
        
        # ATR Volatilité relative
        df['atr_relative'] = df['atr_14'] / df['atr_14'].rolling(window=100).mean()
        df['high_volatility'] = np.where(df['atr_relative'] > 1.5, 1, 0)
        df['low_volatility'] = np.where(df['atr_relative'] < 0.5, 1, 0)
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        """
        Ajoute les indicateurs de volume au DataFrame
        """
        close_price = df['close'].values
        volume = df['volume'].values
        
        # Volume Moving Average
        df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
        
        # Volume relatif
        df['volume_ratio'] = volume / df['volume_sma_20']
        df['high_volume'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
        
        # On-Balance Volume (OBV)
        df['obv'] = talib.OBV(close_price, volume)
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            volume,
            fastperiod=3,
            slowperiod=10
        )
        
        # Money Flow Index
        df['mfi'] = talib.MFI(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            volume,
            timeperiod=14
        )
        
        # Volume Price Trend
        df['vpt'] = df['close'].pct_change() * volume
        df['vpt'] = df['vpt'].cumsum()
        
        return df
    
    @staticmethod
    def add_pivot_points(df):
        """
        Ajoute les points pivots au DataFrame
        """
        # Calcul des points pivots traditionnels (basés sur la période précédente)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['r3'] = df['high'].shift(1) + 2 * (df['pivot'] - df['low'].shift(1))
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        df['s3'] = df['low'].shift(1) - 2 * (df['high'].shift(1) - df['pivot'])
        
        # Support/Résistance dynamique (fractals)
        def find_local_maxima(series, window):
            return (series == series.rolling(window=window, center=True).max())
            
        def find_local_minima(series, window):
            return (series == series.rolling(window=window, center=True).min())
        
        df['fractal_resistance'] = find_local_maxima(df['high'], 5).astype(int)
        df['fractal_support'] = find_local_minima(df['low'], 5).astype(int)
        
        return df
    
    @staticmethod
    def add_candlestick_patterns(df):
        """
        Ajoute les patterns de chandeliers au DataFrame
        """
        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # Patterns de renversement haussier
        df['hammer'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
        df['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_price, high_price, low_price, close_price)
        df['morning_star'] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
        df['piercing'] = talib.CDLPIERCING(open_price, high_price, low_price, close_price)
        df['bullish_engulfing'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
        df['bullish_engulfing'] = np.where(df['bullish_engulfing'] > 0, 1, 0)
        
        # Patterns de renversement baissier
        df['hanging_man'] = talib.CDLHANGINGMAN(open_price, high_price, low_price, close_price)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high_price, low_price, close_price)
        df['evening_star'] = talib.CDLEVENINGSTAR(open_price, high_price, low_price, close_price)
        df['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(open_price, high_price, low_price, close_price)
        df['bearish_engulfing'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
        df['bearish_engulfing'] = np.where(df['bearish_engulfing'] < 0, 1, 0)
        
        # Doji et autres patterns neutres
        df['doji'] = talib.CDLDOJI(open_price, high_price, low_price, close_price)
        df['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open_price, high_price, low_price, close_price)
        df['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(open_price, high_price, low_price, close_price)
        
        # Patterns de continuation
        df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high_price, low_price, close_price)
        df['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high_price, low_price, close_price)
        
        # Agréger les patterns bullish et bearish
        # 1 = pattern bullish, -1 = pattern bearish, 0 = pas de pattern
        bullish_patterns = [
            'hammer', 'inverted_hammer', 'morning_star', 'piercing', 
            'bullish_engulfing', 'three_white_soldiers'
        ]
        
        bearish_patterns = [
            'hanging_man', 'shooting_star', 'evening_star', 'dark_cloud_cover', 
            'bearish_engulfing', 'three_black_crows'
        ]
        
        df['bullish_pattern'] = df[bullish_patterns].sum(axis=1).clip(0, 1)
        df['bearish_pattern'] = df[bearish_patterns].sum(axis=1).clip(0, 1)
        
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    import ccxt
    import matplotlib.pyplot as plt
    
    # Récupérer des données d'exemple
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Ajouter les indicateurs
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Afficher les résultats
    print(df.head())
    print(f"Nombre de colonnes: {len(df.columns)}")
    
    # Tracer un graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Prix')
    plt.plot(df.index, df['ema_20'], label='EMA 20')
    plt.plot(df.index, df['ema_50'], label='EMA 50')
    plt.plot(df.index, df['bb_upper'], 'r--', label='Bollinger Upper')
    plt.plot(df.index, df['bb_lower'], 'r--', label='Bollinger Lower')
    plt.title('Prix et indicateurs')
    plt.legend()
    plt.show()