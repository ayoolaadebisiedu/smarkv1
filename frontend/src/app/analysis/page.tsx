"use client"
import React, { useState, useEffect, useCallback } from 'react';
import { SignalCard } from '@/components/SignalCard';
import { ChartComponent } from '@/components/ChartComponent';
import { TradeModal } from '@/components/TradeModal';
import { TradeHistory } from '@/components/TradeHistory';
import { LayoutDashboard, Bell, Settings, PieChart, Activity, Info, Coins, BarChart4, TrendingUp, TrendingDown, XCircle, Wallet } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

const ASSETS = {
    STOCKS: [
        { name: 'Apple', ticker: 'AAPL' },
        { name: 'Amazon', ticker: 'AMZN' },
        { name: 'Tesla', ticker: 'TSLA' },
        { name: 'Microsoft', ticker: 'MSFT' },
        { name: 'Nvidia', ticker: 'NVDA' },
    ],
    CRYPTO: [
        { name: 'Bitcoin', ticker: 'BTC-USD' },
        { name: 'Ethereum', ticker: 'ETH-USD' },
        { name: 'BNB', ticker: 'BNB-USD' },
        { name: 'Doge', ticker: 'DOGE-USD' },
        { name: 'Solana', ticker: 'SOL-USD' },
    ]
};

export default function AnalysisPage() {
    const [mounted, setMounted] = useState(false);
    const [category, setCategory] = useState<'STOCKS' | 'CRYPTO'>('CRYPTO');
    const [selectedAsset, setSelectedAsset] = useState(ASSETS.CRYPTO[0]);
    const [signals, setSignals] = useState<any[]>([]);
    const [chartData, setChartData] = useState<any[]>([]);
    const [isScanning, setIsScanning] = useState(false);

    // Simulation State
    const [accountSummary, setAccountSummary] = useState({ balance: 10000.0, total_pnl: 0, trades_count: 0 });
    const [activeTrades, setActiveTrades] = useState<any[]>([]);
    const [tradeHistory, setTradeHistory] = useState<any[]>([]);
    const [isTrading, setIsTrading] = useState(false);

    // Modal State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalDirection, setModalDirection] = useState<'buy' | 'sell'>('buy');

    const fetchSignals = async () => {
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/signals`);
            const data = await res.json();
            setSignals(data);
        } catch (e) {
            console.error("Failed to fetch signals", e);
        }
    };

    const fetchHistory = useCallback(async (ticker: string) => {
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            setChartData([]);
            const res = await fetch(`${API_URL}/history/${ticker}`);
            const data = await res.json();
            const formatted = data.map((d: any) => ({
                time: d.time,
                value: d.close,
            }));
            setChartData(formatted);
        } catch (e) {
            console.error("Failed to fetch history", e);
        }
    }, []);

    const fetchAccount = useCallback(async () => {
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/account/summary`);
            const data = await res.json();
            setAccountSummary(data);
        } catch (e) { console.error(e); }
    }, []);

    const fetchActiveTrades = useCallback(async () => {
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/trades/active`);
            const data = await res.json();
            setActiveTrades(data);
        } catch (e) { console.error(e); }
    }, []);

    const fetchTradeHistory = useCallback(async () => {
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/trades/history`);
            const data = await res.json();
            setTradeHistory(data);
        } catch (e) { console.error(e); }
    }, []);

    const openTrade = async (data: { amount: number; sl?: number; tp?: number }) => {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        setIsTrading(true);
        try {
            await fetch(`${API_URL}/trades/open`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: selectedAsset.ticker,
                    direction: modalDirection,
                    amount: data.amount,
                    stop_loss: data.sl,
                    take_profit: data.tp
                })
            });
            await fetchActiveTrades();
            await fetchAccount();
            setIsModalOpen(false);
        } finally {
            setIsTrading(false);
        }
    };

    const closeTrade = async (id: number) => {
        try {
            await fetch(`http://localhost:8000/trades/close/${id}`, { method: 'POST' });
            await fetchActiveTrades();
            await fetchTradeHistory();
            await fetchAccount();
        } catch (e) { console.error(e); }
    };

    const refreshData = async () => {
        setIsScanning(true);
        try {
            await fetch(`http://localhost:8000/scan/${selectedAsset.ticker}`, { method: 'POST' });
            await fetchSignals();
            await fetchHistory(selectedAsset.ticker);
        } finally {
            setIsScanning(false);
        }
    };

    useEffect(() => {
        setMounted(true);
        fetchSignals();
        fetchHistory(selectedAsset.ticker);
        fetchAccount();
        fetchActiveTrades();
        fetchTradeHistory();

        // Auto-refresh account & trades
        const interval = setInterval(() => {
            fetchAccount();
            fetchActiveTrades();
            fetchTradeHistory();
        }, 10000);
        return () => clearInterval(interval);
    }, [selectedAsset.ticker, fetchHistory, fetchAccount, fetchActiveTrades, fetchTradeHistory]);

    if (!mounted) return null;

    return (
        <main className="min-h-screen bg-[#020617] text-slate-200 pt-20">
            <div className="md:ml-20 p-4 md:p-8">
                {/* Section A: Unified Asset Selection Dropdown */}
                <div className="mb-8 flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900/60 p-6 rounded-3xl border border-slate-800 shadow-2xl backdrop-blur-xl">
                    <div className="flex items-center gap-4 w-full md:w-auto">
                        <div className="p-3 bg-blue-600/10 rounded-2xl text-blue-500 border border-blue-500/20">
                            <LayoutDashboard size={24} />
                        </div>
                        <div>
                            <h2 className="text-xl font-black text-white uppercase tracking-tighter">Terminal Selection</h2>
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Select Asset to Analyze</p>
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-4 w-full md:w-auto">
                        {/* Account Summary (Compact in A) */}
                        <div className="hidden lg:flex items-center gap-6 bg-slate-950/50 px-4 py-2 rounded-2xl border border-slate-800/50">
                            <div className="flex flex-col">
                                <span className="text-[8px] uppercase text-slate-500 font-bold tracking-widest leading-none mb-1">Equity</span>
                                <span className="text-white font-mono text-sm leading-none">${accountSummary.balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                            </div>
                            <div className="w-[1px] h-6 bg-slate-800" />
                            <div className="flex flex-col">
                                <span className="text-[8px] uppercase text-slate-500 font-bold tracking-widest leading-none mb-1">Realized P&L</span>
                                <span className={cn(
                                    "font-mono text-sm leading-none",
                                    accountSummary.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                )}>
                                    {accountSummary.total_pnl >= 0 ? '+' : ''}${accountSummary.total_pnl.toFixed(2)}
                                </span>
                            </div>
                        </div>

                        {/* Unified Dropdown Selection */}
                        <div className="relative group w-full md:w-64">
                            <select
                                value={`${category}:${selectedAsset.ticker}`}
                                onChange={(e) => {
                                    const [newCat, newTicker] = e.target.value.split(':') as ['STOCKS' | 'CRYPTO', string];
                                    setCategory(newCat);
                                    const asset = (ASSETS[newCat] as any[]).find(a => a.ticker === newTicker);
                                    if (asset) setSelectedAsset(asset);
                                }}
                                className="w-full bg-slate-950 border border-slate-800 text-slate-200 px-4 py-3 rounded-2xl appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all font-bold text-sm tracking-tight"
                            >
                                <optgroup label="CRYPTO" className="bg-slate-950 text-slate-400 font-black uppercase tracking-widest text-[10px]">
                                    {ASSETS.CRYPTO.map(a => (
                                        <option key={a.ticker} value={`CRYPTO:${a.ticker}`} className="text-slate-200 py-2">
                                            {a.name} ({a.ticker})
                                        </option>
                                    ))}
                                </optgroup>
                                <optgroup label="STOCKS" className="bg-slate-950 text-slate-400 font-black uppercase tracking-widest text-[10px]">
                                    {ASSETS.STOCKS.map(a => (
                                        <option key={a.ticker} value={`STOCKS:${a.ticker}`} className="text-slate-200 py-2">
                                            {a.name} ({a.ticker})
                                        </option>
                                    ))}
                                </optgroup>
                            </select>
                            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 group-hover:text-blue-400 transition-colors">
                                <Activity size={16} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex flex-col xl:flex-row gap-8 relative items-start">
                    {/* Section B: Static Sidebar (Analysis Suggestions) */}
                    <aside className="xl:w-[380px] space-y-8 xl:sticky xl:top-28 w-full">
                        {/* Live Feed Panel */}
                        <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 shadow-xl backdrop-blur-sm">
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-xl font-black text-white flex items-center gap-2 uppercase tracking-tighter">
                                    Live Feed
                                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                                </h2>
                                <button onClick={fetchSignals} className="text-blue-500 text-[10px] font-bold hover:underline uppercase tracking-widest">Reload</button>
                            </div>
                            <div className="space-y-4 max-h-[350px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-800">
                                {signals.length === 0 ? (
                                    <p className="text-slate-500 text-center py-10 uppercase text-[10px] font-bold tracking-widest">Scanning Meta-Data...</p>
                                ) : (
                                    signals.map((signal, idx) => (
                                        <SignalCard key={idx} signal={signal} />
                                    ))
                                )}
                            </div>
                        </div>

                        {/* Insight Grid (Moved to static section B) */}
                        <div className="space-y-4">
                            <h2 className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] px-2 mb-4">Risk & Logic Intelligence</h2>

                            <div className="bg-slate-900/40 p-5 rounded-3xl border border-slate-800 shadow-lg hover:border-blue-500/20 transition-colors group">
                                <div className="flex items-center gap-3 mb-3 text-blue-400">
                                    <Activity size={18} />
                                    <h4 className="font-black uppercase tracking-widest text-[10px] text-white">Sentiment Alpha</h4>
                                </div>
                                <p className="text-xs text-slate-400 leading-relaxed font-medium">
                                    Scanning news for institutional word-choice shifts on {selectedAsset.ticker}.
                                </p>
                            </div>

                            <div className="bg-slate-900/40 p-5 rounded-3xl border border-slate-800 shadow-lg hover:border-emerald-500/20 transition-colors group">
                                <div className="flex items-center gap-3 mb-3 text-emerald-400">
                                    <Info size={18} />
                                    <h4 className="font-black uppercase tracking-widest text-[10px] text-white">Convergence Logic</h4>
                                </div>
                                <p className="text-xs text-slate-400 leading-relaxed font-medium">
                                    MACD/EMA filters suggesting {signals.some(s => s.type.includes('Bullish')) ? 'Bullish' : 'Neutral'} momentum bias.
                                </p>
                            </div>

                            <div className="bg-slate-900/40 p-5 rounded-3xl border border-slate-800 shadow-lg hover:border-amber-500/20 transition-colors group">
                                <div className="flex items-center gap-3 mb-3 text-amber-400">
                                    <Settings size={18} />
                                    <h4 className="font-black uppercase tracking-widest text-[10px] text-white">Risk Intelligence</h4>
                                </div>
                                <p className="text-xs text-slate-400 leading-relaxed font-medium">
                                    Every signal is filtered through calculated probability models before ingestion.
                                </p>
                            </div>
                        </div>

                        {/* Active Trades Panel (Compact in Sidebar) */}
                        <div className="bg-slate-950/40 border border-slate-800/50 rounded-3xl p-6">
                            <h2 className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-6 flex items-center justify-between">
                                Active Positions
                                <span className="bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded-full text-[8px]">{activeTrades.length} Total</span>
                            </h2>
                            <div className="space-y-3">
                                {activeTrades.length === 0 ? (
                                    <div className="text-center py-6 grayscale opacity-30">
                                        <p className="text-[10px] font-bold uppercase tracking-widest">No active exposure</p>
                                    </div>
                                ) : (
                                    activeTrades.slice(0, 3).map((trade) => (
                                        <div key={trade.id} className="p-3 bg-slate-900 border border-slate-800 rounded-xl group flex justify-between items-center transition-all hover:bg-slate-800/50">
                                            <div className="space-y-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-bold text-white uppercase text-[10px]">{trade.ticker}</span>
                                                    <span className={cn(
                                                        "text-[8px] px-1.5 py-0.5 rounded-md font-black uppercase",
                                                        trade.direction === 'buy' ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"
                                                    )}>
                                                        {trade.direction === 'buy' ? 'LONG' : 'SHORT'}
                                                    </span>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => closeTrade(trade.id)}
                                                className="p-1.5 text-slate-600 hover:text-rose-500 transition-colors"
                                            >
                                                <XCircle size={14} />
                                            </button>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </aside>

                    {/* Section C: Main Scrolling Content */}
                    <section className="flex-1 space-y-8 min-w-0">
                        {/* Main Interaction Panel */}
                        <div className="bg-slate-900/40 rounded-3xl p-6 md:p-8 border border-slate-800 shadow-2xl relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/5 blur-[120px] -z-10" />

                            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-6">
                                <div>
                                    <div className="flex items-center gap-3 mb-1">
                                        <h3 className="text-3xl font-extrabold text-white uppercase tracking-tighter">{selectedAsset.ticker}</h3>
                                        <div className="h-4 w-[1px] bg-slate-800" />
                                        <span className="text-emerald-400 font-mono text-xs flex items-center gap-1.5 px-2 py-1 bg-emerald-400/5 border border-emerald-400/10 rounded-lg">
                                            <TrendingUp size={12} /> LIVE ENGINE
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-500 font-bold uppercase tracking-wider">{selectedAsset.name} Terminal Analysis</p>
                                </div>

                                <div className="flex flex-wrap gap-3">
                                    <button
                                        onClick={() => {
                                            setModalDirection('buy');
                                            setIsModalOpen(true);
                                        }}
                                        disabled={isTrading}
                                        className="px-6 py-3 bg-emerald-600/10 text-emerald-500 border border-emerald-500/30 rounded-2xl text-xs font-black hover:bg-emerald-600 hover:text-white transition-all flex items-center gap-2 uppercase group"
                                    >
                                        <TrendingUp size={16} className="group-hover:scale-110 transition-transform" /> Market Buy
                                    </button>
                                    <button
                                        onClick={() => {
                                            setModalDirection('sell');
                                            setIsModalOpen(true);
                                        }}
                                        disabled={isTrading}
                                        className="px-6 py-3 bg-rose-600/10 text-rose-500 border border-rose-500/30 rounded-2xl text-xs font-black hover:bg-rose-600 hover:text-white transition-all flex items-center gap-2 uppercase group"
                                    >
                                        <TrendingDown size={16} className="group-hover:scale-110 transition-transform" /> Market Short
                                    </button>

                                    <div className="w-[1px] h-10 bg-slate-800 mx-1 hidden md:block" />

                                    <button
                                        onClick={refreshData}
                                        disabled={isScanning}
                                        className={cn(
                                            "px-8 py-3 rounded-2xl text-xs font-black transition-all flex items-center gap-2 uppercase shadow-xl shadow-blue-500/10",
                                            isScanning
                                                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                                                : "bg-blue-600 text-white hover:bg-blue-500"
                                        )}
                                    >
                                        {isScanning ? <Activity className="animate-spin" size={16} /> : <Activity size={16} />}
                                        {isScanning ? "Scanning..." : "Scan Market"}
                                    </button>
                                </div>
                            </div>

                            <div className="h-[500px] w-full rounded-2xl overflow-hidden border border-slate-800/50 bg-slate-950/40">
                                {chartData.length > 0 ? (
                                    <ChartComponent data={chartData} />
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
                                        <div className="w-10 h-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin" />
                                        <p className="font-bold animate-pulse uppercase tracking-[0.3em] text-[10px] text-blue-500/50">Synchronizing Analysis Engine...</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Trade History (Main scroll area) */}
                        <div className="bg-slate-900/40 rounded-3xl p-1 border border-slate-800 overflow-hidden shadow-xl">
                            <TradeHistory trades={tradeHistory} />
                        </div>
                    </section>
                </div>
            </div>

            {/* Trade Modal */}
            <TradeModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                asset={selectedAsset}
                direction={modalDirection}
                onConfirm={openTrade}
                currentPrice={chartData.length > 0 ? chartData[chartData.length - 1].value : 0}
                suggestedSL={signals.find(s => s.ticker === selectedAsset.ticker)?.sl}
                suggestedTP={signals.find(s => s.ticker === selectedAsset.ticker)?.tp}
            />
        </main>
    );
}
