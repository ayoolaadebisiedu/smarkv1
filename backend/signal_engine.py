import pandas as pd
import pandas_ta as ta
import numpy as np
from .algo_suite import fetch_real_sentiment, detect_turtle_breakout, detect_ichimoku_signals

def detect_divergence(df, lookback=5):
    """
    Detects RSI and OBV Divergence.
    """
    if len(df) < 50:
        return []

    df = df.copy()
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df = df.dropna().reset_index(drop=True)

    if len(df) < 2 * lookback + 1:
        return []

    signals = []
    
    # helper to find peaks/troughs
    def is_trough(idx, col):
        if idx < lookback or idx > len(df) - lookback - 1: return False
        return df[col].iloc[idx] == df[col].iloc[idx-lookback:idx+lookback+1].min()

    def is_peak(idx, col):
        if idx < lookback or idx > len(df) - lookback - 1: return False
        return df[col].iloc[idx] == df[col].iloc[idx-lookback:idx+lookback+1].max()

    # Scan for recent RSI Divergence
    i = len(df) - lookback - 1
    if is_trough(i, 'low'):
        # Find previous trough
        prev_i = -1
        for j in range(i - lookback, lookback, -1):
            if is_trough(j, 'low'):
                prev_i = j
                break
        
        if prev_i != -1:
            if df['low'].iloc[i] < df['low'].iloc[prev_i] and df['RSI'].iloc[i] > df['RSI'].iloc[prev_i]:
                signals.append({"type": "Regular Bullish RSI Divergence", "confidence": 85, "entry_price": float(df['close'].iloc[-1]), "indicator": "RSI"})
    
    if is_peak(i, 'high'):
        prev_i = -1
        for j in range(i - lookback, lookback, -1):
            if is_peak(j, 'high'):
                prev_i = j
                break
        if prev_i != -1:
            if df['high'].iloc[i] > df['high'].iloc[prev_i] and df['RSI'].iloc[i] < df['RSI'].iloc[prev_i]:
                signals.append({"type": "Regular Bearish RSI Divergence", "confidence": 88, "entry_price": float(df['close'].iloc[-1]), "indicator": "RSI"})

    return signals

def detect_bull_flag(df):
    """
    Strategy: 15m Momentum Pole (spike) followed by tight consolidation (flag).
    """
    if len(df) < 20: return []
    
    # 1. Detect the "Pole" (spike in price and volume)
    # Look back 5-10 bars for a significant move
    recent_returns = df['close'].pct_change(5).iloc[-10:-2]
    max_return = recent_returns.max()
    
    if max_return > 0.02: # 2% move in 5 bars (customizable)
        # 2. Detect the "Flag" (consolidation)
        # Recent 3-5 bars should be in a tight range or slight pullback
        flag_bars = df.iloc[-5:]
        flag_top = flag_bars['high'].max()
        flag_bottom = flag_bars['low'].min()
        pole_top = df['high'].iloc[-10:-5].max()
        
        # Consolidation check: price staying in top 50% of the movement
        if flag_bottom > (df['close'].iloc[-10] + (pole_top - df['close'].iloc[-10]) * 0.5):
            return [{
                "type": "Bull Flag Formation",
                "confidence": 82,
                "strategy": "Momentum Breakout",
                "reasoning": f"Detected {max_return*100:.1f}% pole followed by tight consolidation.",
                "entry": float(flag_top * 1.001),
                "sl": float(flag_bottom * 0.995),
                "tp": float(flag_top + (pole_top - df['close'].iloc[-10]))
            }]
    return []

def detect_double_bottom(df, lookback=20):
    """
    Strategy: "W" pattern at potential support.
    """
    if len(df) < 50: return []
    
    lows = df['low'].rolling(window=5, center=True).min()
    potential_troughs = []
    for i in range(len(df)-lookback, len(df)-2):
        if df['low'].iloc[i] == lows.iloc[i]:
            potential_troughs.append(i)
            
    if len(potential_troughs) >= 2:
        t1, t2 = potential_troughs[-2], potential_troughs[-1]
        price1, price2 = df['low'].iloc[t1], df['low'].iloc[t2]
        
        # Check if they are roughly at the same level (within 0.5%)
        if abs(price1 - price2) / price1 < 0.005:
            # Neckline is the peak between the two troughs
            neckline = df['high'].iloc[t1:t2].max()
            current_price = df['close'].iloc[-1]
            
            if current_price > neckline * 0.95: # Close to or breaking neckline
                return [{
                    "type": "Double Bottom (W-Pattern)",
                    "confidence": 90,
                    "strategy": "Support Reversal",
                    "reasoning": "Asset found support twice at similar levels. High probability bounce.",
                    "entry": float(neckline * 1.002),
                    "sl": float(min(price1, price2) * 0.99),
                    "tp": float(neckline + (neckline - min(price1, price2)))
                }]
    return []

def detect_macd_cross(df):
    if len(df) < 200: return []
    df = df.copy()
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['EMA200'] = ta.ema(df['close'], length=200)
    
    signals = []
    hist = df['MACDh_12_26_9']
    
    if (hist.iloc[-2] < 0 and hist.iloc[-1] > 0 and 
        df['close'].iloc[-1] > df['EMA200'].iloc[-1]):
        signals.append({
            "type": "MACD Bullish Cross",
            "confidence": 82,
            "entry_price": float(df['close'].iloc[-1]),
            "indicator": "MACD/EMA200"
        })
    return signals

def generate_pro_analysis(ticker, df):
    """
    Combined analysis targeted at specific "Titan" strategies.
    """
    # 1. Ticker specific logic
    if any(sym in ticker for sym in ['TSLA', 'BTC', 'SOL']):
        # Momentum Assets -> Bull Flags
        flags = detect_bull_flag(df)
        if flags: return flags[0]
        
    if any(sym in ticker for sym in ['AAPL', 'MSFT', 'AMZN']):
        # Defensive/Reliable -> Double Bottoms
        w_pattern = detect_double_bottom(df)
        if w_pattern: return w_pattern[0]
        
    # 2. Try Turtle Strategy (Institutional Trend Following)
    turtle = detect_turtle_breakout(df, system=2) # Prefer 55-day for high conviction
    if not turtle:
        turtle = detect_turtle_breakout(df, system=1)
    if turtle: return turtle[0]
        
    # Catch-all: Divergence
    divs = detect_divergence(df)
    if divs:
        d = divs[0]
        # Calculate SL/TP for divergence
        entry = d['entry_price']
        if "Bullish" in d['type']:
            sl = entry * 0.98
            tp = entry * 1.05
        else:
            sl = entry * 1.02
            tp = entry * 0.95
            
        return {
            "type": d['type'],
            "confidence": d['confidence'],
            "strategy": "Mean Reversion",
            "reasoning": f"Price and {d['indicator']} are moving in opposite directions. Expect snapback.",
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp)
        }
            
    return None

def detect_sentiment(ticker: str):
    # Phase 1: Use REAL Sentiment from algo_suite
    try:
        real_sent = fetch_real_sentiment(ticker)
        if real_sent:
            return [{
                "type": real_sent[0]["type"],
                "confidence": real_sent[0]["confidence"],
                "entry_price": 0.0,
                "indicator": "VADER/Google News",
                "reasoning": real_sent[0]["reasoning"]
            }]
    except Exception as e:
        print(f"Sentiment fetch failed: {e}")
        
    # Fallback to local logic (mock)
    mock_headlines = {
        "BTC-USD": ["Bitcoin surges as ETF inflows hit record highs", "Adoption of BTC increasing in emerging markets"],
        "TSLA": ["Tesla delivery numbers beat expectations", "New Gigafactory expansion announced"],
        "AAPL": ["Apple iPhone sales in China showing resilience", "New AI features to boost iPad demand"]
    }
    headlines = mock_headlines.get(ticker, ["Market remains cautious ahead of central bank meeting"])
    positive_words = ["surge", "higher", "growth", "positive", "strong", "bull", "earnings", "hit", "beat", "expansion"]
    negative_words = ["pressure", "lower", "weak", "cut", "negative", "bear", "drag", "drop", "warn"]
    score = 0
    for headline in headlines:
        for word in positive_words:
            if word in headline.lower(): score += 1
        for word in negative_words:
            if word in headline.lower(): score -= 1
    if score > 0:
        return [{"type": "Bullish News Sentiment", "confidence": 75 + (score * 5), "entry_price": 0.0, "indicator": "News Scanner"}]
    elif score < 0:
        return [{"type": "Bearish News Sentiment", "confidence": 75 + (abs(score) * 5), "entry_price": 0.0, "indicator": "News Scanner"}]
    return []
