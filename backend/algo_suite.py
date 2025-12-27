import pandas as pd
import pandas_ta as ta
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote

# Phase 1: Real-time News Sentiment Logic (VADER + Google News)
def fetch_real_sentiment(ticker: str):
    """
    Scrapes Google News RSS for the ticker and performs VADER sentiment analysis.
    """
    search_query = f"{ticker} stock price market news"
    if "USD" in ticker:
        search_query = f"{ticker.replace('-USD', '')} crypto price news"
        
    rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}"
    feed = feedparser.parse(rss_url)
    headlines = [item.title for item in feed.entries[:10]]
    
    if not headlines:
        return []

    analyzer = SentimentIntensityAnalyzer()
    total_score = 0
    valid_headlines = 0
    
    for title in headlines:
        score = analyzer.polarity_scores(title)['compound']
        total_score += score
        valid_headlines += 1
        
    if valid_headlines == 0:
        return []
        
    avg_score = total_score / valid_headlines
    
    if avg_score > 0.1:
        return [{"type": "Institutional Bullish Sentiment", "confidence": int(70 + avg_score * 30), "reasoning": f"Analyzed {valid_headlines} news sources. VADER score: {avg_score:.2f}"}]
    elif avg_score < -0.1:
        return [{"type": "Institutional Bearish Sentiment", "confidence": int(70 + abs(avg_score) * 30), "reasoning": f"Analyzed {valid_headlines} news sources. VADER score: {avg_score:.2f}"}]
    
    return []

# Github Request: Turtle Trading Strategy logic
def detect_turtle_breakout(df: pd.DataFrame, system: int = 1):
    """
    Implements the core Turtle Trading breakout logic.
    System 1: 20-day high/low.
    System 2: 55-day high/low.
    """
    if len(df) < 60:
        return []

    df = df.copy()
    
    # 1. Compute Donchian Channels
    if system == 1:
        entry_window = 20
        exit_window = 10
    else:
        entry_window = 55
        exit_window = 20
        
    df['donchian_high'] = df['high'].shift(1).rolling(window=entry_window).max()
    df['donchian_low'] = df['low'].shift(1).rolling(window=entry_window).min()
    df['exit_high'] = df['high'].shift(1).rolling(window=exit_window).max()
    df['exit_low'] = df['low'].shift(1).rolling(window=exit_window).min()
    
    # 2. Compute ATR (N) for stops
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=20)
    
    current_price = df['close'].iloc[-1]
    prev_high = df['donchian_high'].iloc[-1]
    prev_low = df['donchian_low'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    if pd.isna(prev_high) or pd.isna(atr):
        return []

    # Entry Signal
    if current_price > prev_high:
        return [{
            "type": f"Turtle System {system} Long breakout",
            "strategy": "Trend Following",
            "confidence": 85 if system == 1 else 92,
            "reasoning": f"Price broke above {entry_window}-day resistance level of ${prev_high:.2f}",
            "entry": float(current_price * 1.001),
            "sl": float(current_price - (2 * atr)),
            "tp": float(current_price + (4 * atr)) # Dynamic TP for walkthrough
        }]
    elif current_price < prev_low:
        return [{
            "type": f"Turtle System {system} Short breakout",
            "strategy": "Trend Following",
            "confidence": 85 if system == 1 else 92,
            "reasoning": f"Price broke below {entry_window}-day support level of ${prev_low:.2f}",
            "entry": float(current_price * 0.999),
            "sl": float(current_price + (2 * atr)),
            "tp": float(current_price - (4 * atr))
        }]

    return []

# Phase 2: Ichimoku Logic (Placeholder for full integration)
def detect_ichimoku_signals(df: pd.DataFrame):
    if len(df) < 52: return []
    # Simplified Ichimoku Tenkan/Kijun cross
    tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    
    if tenkan.iloc[-2] <= kijun.iloc[-2] and tenkan.iloc[-1] > kijun.iloc[-1]:
        return [{"type": "Ichimoku Bullish Cross", "confidence": 78, "indicator": "Tenkan/Kijun"}]
    elif tenkan.iloc[-2] >= kijun.iloc[-2] and tenkan.iloc[-1] < kijun.iloc[-1]:
        return [{"type": "Ichimoku Bearish Cross", "confidence": 78, "indicator": "Tenkan/Kijun"}]
    return []
