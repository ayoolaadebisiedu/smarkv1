import pandas as pd
import pandas_ta as ta
import numpy as np

def detect_divergence(df, lookback=5):
    """
    df: DataFrame with ['open', 'high', 'low', 'close']
    lookback: number of candles to confirm a local peak/trough
    """
    if len(df) < 50:  # Need enough data for RSI and lookback
        return []

    # 1. Calculate RSI
    df = df.copy()
    df['RSI'] = ta.rsi(df['close'], length=14)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 2 * lookback + 1:
        return []

    # 2. Find Local Minima (Troughs) for Bullish Divergence
    # A trough is a point lower than 'lookback' points before and after it
    df['price_min'] = df['low'][(df['low'] == df['low'].rolling(2 * lookback + 1, center=True).min())]
    df['rsi_min'] = df['RSI'][(df['RSI'] == df['RSI'].rolling(2 * lookback + 1, center=True).min())]

    signals = []
    
    # Logic for scanning troughs
    troughs = df.dropna(subset=['price_min', 'rsi_min'])
    if len(troughs) >= 2:
        prev_trough = troughs.iloc[-2]
        curr_trough = troughs.iloc[-1]
        
        # Regular Bullish Divergence: Price makes Lower Low (LL), RSI makes Higher Low (HL)
        if curr_trough['price_min'] < prev_trough['price_min'] and \
           curr_trough['rsi_min'] > prev_trough['rsi_min']:
            signals.append({
                "type": "Regular Bullish Divergence",
                "confidence": 85,
                "entry_price": float(df['close'].iloc[-1]),
                "indicator": "RSI"
            })
            
        # Hidden Bullish Divergence: Price makes Higher Low (HL), RSI makes Lower Low (LL)
        elif curr_trough['price_min'] > prev_trough['price_min'] and \
             curr_trough['rsi_min'] < prev_trough['rsi_min']:
            signals.append({
                "type": "Hidden Bullish Divergence",
                "confidence": 75,
                "entry_price": float(df['close'].iloc[-1]),
                "indicator": "RSI"
            })

    return signals

def detect_macd_cross(df):
    """
    df: DataFrame with ['open', 'high', 'low', 'close']
    Returns MACD signals based on historical patterns.
    """
    if len(df) < 200: # Need enough data for EMA 200
        return []

    df = df.copy()
    
    # Calculate MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # Calculate EMA 200 for trend filter
    df['EMA200'] = ta.ema(df['close'], length=200)
    
    signals = []
    
    # Column names from pandas_ta for MACD: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    # MACDh is the histogram (MACD - Signal)
    hist = df['MACDh_12_26_9']
    macd_val = df['MACD_12_26_9']
    signal_val = df['MACDs_12_26_9']
    
    # Check for cross on the latest completed candle
    if (hist.iloc[-2] < 0 and hist.iloc[-1] > 0 and 
        macd_val.iloc[-1] < 0 and signal_val.iloc[-1] < 0 and
        df['close'].iloc[-1] > df['EMA200'].iloc[-1]):
        
        signals.append({
            "type": "MACD Bullish Cross",
            "confidence": 82,
            "entry_price": float(df['close'].iloc[-1]),
            "indicator": "MACD/EMA200"
        })
        
    return signals

def detect_sentiment(ticker: str):
    """
    Scans recent news for the ticker and returns a sentiment-based signal.
    Placeholder for actual news fetching and VADER/FinBERT analysis.
    """
    # Mock news headlines for demonstration
    mock_headlines = {
        "BTCUSDT": ["Bitcoin surges as ETF inflows hit record highs", "Adoption of BTC increasing in emerging markets"],
        "EURUSD": ["Euro under pressure as ECB hints at rate cuts", "Weak manufacturing data from Germany drags Euro"],
        "AMZN": ["Amazon reports better than expected earnings", "AWS growth continues to outpace competitors"]
    }
    
    headlines = mock_headlines.get(ticker, ["Market remains cautious ahead of central bank meeting"])
    
    # Simple keyword-based sentiment for now
    positive_words = ["surge", "higher", "growth", "positive", "strong", "bull", "earnings", "hit"]
    negative_words = ["pressure", "lower", "weak", "cut", "negative", "bear", "drag", "drop"]
    
    score = 0
    for headline in headlines:
        for word in positive_words:
            if word in headline.lower():
                score += 1
        for word in negative_words:
            if word in headline.lower():
                score -= 1
                
    if score > 0:
        return [{
            "type": "Bullish News Sentiment",
            "confidence": 70 + (score * 5),
            "entry_price": 0.0, # Not applicable for sentiment only
            "indicator": "News Scanner"
        }]
    elif score < 0:
        return [{
            "type": "Bearish News Sentiment",
            "confidence": 70 + (abs(score) * 5),
            "entry_price": 0.0,
            "indicator": "News Scanner"
        }]
    
    return []
