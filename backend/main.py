from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import SessionLocal, init_db, Asset, Signal, Trade, Account
from .signal_engine import detect_divergence, detect_macd_cross, detect_sentiment
from .risk_manager import RiskManager
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import random
from datetime import datetime, timedelta

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smark Signal Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    init_db()

class RiskRequest(BaseModel):
    account_balance: float
    risk_per_trade: float = 0.01
    entry_price: float
    stop_loss: float
    pip_value: float = 10.0
    is_forex: bool = True

class TradeOpenRequest(BaseModel):
    ticker: str
    direction: str # buy, sell
    amount: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@app.get("/")
def read_root():
    return {"message": "Smark API is running"}

@app.get("/account/summary")
def get_account_summary(db: Session = Depends(get_db)):
    account = db.query(Account).first()
    if not account:
        account = Account(balance=10000.0)
        db.add(account)
        db.commit()
        db.refresh(account)
    
    closed_trades = db.query(Trade).filter(Trade.status == "Closed").all()
    total_pnl = sum(t.pnl for t in closed_trades)
    
    return {
        "balance": account.balance,
        "total_pnl": total_pnl,
        "trades_count": len(closed_trades)
    }

@app.post("/trades/open")
def open_trade(req: TradeOpenRequest, db: Session = Depends(get_db)):
    df = fetch_live_data(req.ticker, period="1d", interval="1m")
    if df.empty:
        # Fallback if 1m is not available for stocks sometimes after hours
        df = fetch_live_data(req.ticker, period="5d", interval="1h")
        if df.empty:
            raise HTTPException(status_code=400, detail="Could not fetch current price")
    
    current_price = float(df['close'].iloc[-1])
    
    trade = Trade(
        ticker=req.ticker,
        direction=req.direction,
        entry_price=current_price,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        amount=req.amount,
        status="Open"
    )
    db.add(trade)
    db.commit()
    return {"message": "Trade opened", "entry_price": current_price}

@app.get("/trades/active")
def get_active_trades(db: Session = Depends(get_db)):
    trades = db.query(Trade).filter(Trade.status == "Open").all()
    return [
        {
            "id": t.id,
            "ticker": t.ticker,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "amount": t.amount,
            "created_at": t.created_at
        } for t in trades
    ]

@app.get("/trades/history", response_model=List[dict])
def get_trade_history(db: Session = Depends(get_db)):
    trades = db.query(Trade).filter(Trade.status == "Closed").order_by(Trade.closed_at.desc()).all()
    return [
        {
            "id": t.id,
            "ticker": t.ticker,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "amount": t.amount,
            "pnl": t.pnl,
            "status": t.status,
            "created_at": t.created_at,
            "closed_at": t.closed_at
        } for t in trades
    ]

@app.post("/trades/close/{trade_id}")
def close_trade(trade_id: int, db: Session = Depends(get_db)):
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade or trade.status == "Closed":
        raise HTTPException(status_code=404, detail="Trade not found or already closed")
    
    df = fetch_live_data(trade.ticker, period="1d", interval="1m")
    if df.empty:
        df = fetch_live_data(trade.ticker, period="5d", interval="1h")
        if df.empty:
             raise HTTPException(status_code=400, detail="Could not fetch current price")
    
    exit_price = float(df['close'].iloc[-1])
    
    if trade.direction == "buy":
        pnl = (exit_price - trade.entry_price) * trade.amount
    else:
        pnl = (trade.entry_price - exit_price) * trade.amount
    
    trade.exit_price = exit_price
    trade.pnl = pnl
    trade.status = "Closed"
    trade.closed_at = datetime.utcnow()
    
    account = db.query(Account).first()
    if account:
        account.balance += pnl
    
    db.commit()
    return {"message": "Trade closed", "exit_price": exit_price, "pnl": pnl}

@app.get("/signals", response_model=List[dict])
def get_signals(db: Session = Depends(get_db)):
    signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(20).all()
    return [
        {
            "ticker": s.asset.ticker,
            "type": s.signal_type,
            "confidence": s.confidence,
            "entry": s.entry_price,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "created_at": s.created_at
        } for s in signals
    ]

@app.post("/calculate-risk")
def calculate_risk(req: RiskRequest):
    rm = RiskManager(req.account_balance, req.risk_per_trade)
    pos_size = rm.calculate_position_size(req.entry_price, req.stop_loss, req.pip_value, req.is_forex)
    rr = rm.get_risk_reward_ratio(req.entry_price, req.stop_loss, req.entry_price + (abs(req.entry_price - req.stop_loss) * 2)) # Default 1:2
    return {
        "position_size": pos_size,
        "risk_reward": rr,
        "unit": "Lots" if req.is_forex else "Units"
    }

# Real data fetcher
import yfinance as yf

def fetch_live_data(ticker: str, period="1mo", interval="1h"):
    try:
        yf_ticker = ticker
        if ticker.endswith("USDT"):
             yf_ticker = ticker.replace("USDT", "-USD")
        
        # Using Ticker.history for a cleaner flat DataFrame
        ticker_obj = yf.Ticker(yf_ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
            
        # Normalize columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            # If columns are named differently (e.g. capitalized), this handles it.
            # Ticker.history usually returns Capitalized names.
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@app.get("/history/{ticker}")
def get_history(ticker: str):
    # Retrieve real historical data for charting
    df = fetch_live_data(ticker, period="6mo", interval="1d")
    
    if df.empty:
        return []
        
    # Convert to lightweight-charts format
    chart_data = []
    
    # yf.download with interval='1d' usually returns DatetimeIndex
    for index, row in df.iterrows():
        chart_data.append({
            "time": index.strftime("%Y-%m-%d"),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close']
        })
    return chart_data

# Mock endpoint to trigger signal scan (In production this would be a background task)
@app.post("/scan/{ticker}")
def scan_ticker(ticker: str, db: Session = Depends(get_db)):
    # Fetch Live Data
    df = fetch_live_data(ticker, period="1mo", interval="1h") 
    
    if df.empty or len(df) < 50:
        return {"message": f"Not enough data found for {ticker}", "signals_found": 0}

    signals = detect_divergence(df)
    signals.extend(detect_macd_cross(df))
    signals.extend(detect_sentiment(ticker))
    
    asset = db.query(Asset).filter(Asset.ticker == ticker).first()
    if not asset:
        asset = Asset(ticker=ticker, asset_class="Mock")
        db.add(asset)
        db.commit()
        db.refresh(asset)
        
    for sig in signals:
        # Avoid duplicates logic (omitted for brevity)
        db_sig = Signal(
            asset_id=asset.id,
            signal_type=sig["type"],
            confidence=sig["confidence"],
            entry_price=sig.get("entry_price", 0.0),
            stop_loss=sig.get("entry_price", 0.0) * 0.98, # Mock SL
            take_profit=sig.get("entry_price", 0.0) * 1.05, # Mock TP
        )
        db.add(db_sig)
        
    db.commit()
    return {"message": f"Scan complete for {ticker}", "signals_found": len(signals), "details": signals}
