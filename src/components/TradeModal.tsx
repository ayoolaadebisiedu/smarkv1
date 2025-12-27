"use client"
import React, { useState, useEffect } from 'react';
import { X, TrendingUp, TrendingDown, Info, Shield, Target, DollarSign } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface TradeModalProps {
    isOpen: boolean;
    onClose: () => void;
    asset: { name: string; ticker: string };
    direction: 'buy' | 'sell';
    onConfirm: (data: { amount: number; sl?: number; tp?: number }) => void;
    currentPrice: number;
    suggestedSL?: number;
    suggestedTP?: number;
}

export function TradeModal({ isOpen, onClose, asset, direction, onConfirm, currentPrice, suggestedSL, suggestedTP }: TradeModalProps) {
    const [amount, setAmount] = useState(10);
    const [sl, setSl] = useState<string>(suggestedSL?.toFixed(2) || '');
    const [tp, setTp] = useState<string>(suggestedTP?.toFixed(2) || '');

    useEffect(() => {
        if (suggestedSL) setSl(suggestedSL.toFixed(2));
        if (suggestedTP) setTp(suggestedTP.toFixed(2));
    }, [suggestedSL, suggestedTP]);

    if (!isOpen) return null;

    const rrRatio = (sl && tp) ? (Math.abs(parseFloat(tp) - currentPrice) / Math.abs(currentPrice - parseFloat(sl))).toFixed(2) : 'N/A';

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={onClose} />
            <div className="relative w-full max-w-md bg-[#0f172a] border border-slate-800 rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                {/* Header */}
                <div className={cn(
                    "p-6 border-b border-slate-800 flex justify-between items-center",
                    direction === 'buy' ? "bg-emerald-500/5" : "bg-rose-500/5"
                )}>
                    <div className="flex items-center gap-3">
                        <div className={cn(
                            "p-2 rounded-xl",
                            direction === 'buy' ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"
                        )}>
                            {direction === 'buy' ? <TrendingUp size={24} /> : <TrendingDown size={24} />}
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white uppercase tracking-tight">Open {direction === 'buy' ? 'Long' : 'Short'}</h2>
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">{asset.name} ({asset.ticker})</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-xl text-slate-500 transition-colors">
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-6">
                    {/* Price Info */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-slate-900/50 rounded-2xl border border-slate-800/50">
                            <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest block mb-1">Market Price</span>
                            <span className="text-xl font-mono text-white font-bold">${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                        </div>
                        <div className="p-4 bg-slate-900/50 rounded-2xl border border-slate-800/50">
                            <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest block mb-1">Risk/Reward</span>
                            <span className={cn(
                                "text-xl font-mono font-bold",
                                parseFloat(rrRatio) >= 2 ? "text-emerald-400" : "text-amber-400"
                            )}>{rrRatio}:1</span>
                        </div>
                    </div>

                    {/* Form Controls */}
                    <div className="space-y-4">
                        <div>
                            <label className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1.5 flex items-center gap-2">
                                <DollarSign size={10} /> Position Amount
                            </label>
                            <input
                                type="number"
                                value={amount}
                                onChange={(e) => setAmount(parseFloat(e.target.value))}
                                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-white font-mono focus:outline-none focus:border-blue-500 transition-colors"
                                placeholder="Investment amount"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1.5 flex items-center gap-2">
                                    <Shield size={10} /> Stop Loss
                                </label>
                                <input
                                    type="number"
                                    value={sl}
                                    onChange={(e) => setSl(e.target.value)}
                                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-white font-mono focus:outline-none focus:border-rose-500 transition-colors"
                                    placeholder="Price level"
                                />
                                {suggestedSL && (
                                    <button
                                        onClick={() => setSl(suggestedSL.toFixed(2))}
                                        className="text-[9px] text-rose-500/70 hover:text-rose-500 mt-1 font-bold uppercase"
                                    >
                                        Auto: ${suggestedSL.toFixed(2)}
                                    </button>
                                )}
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1.5 flex items-center gap-2">
                                    <Target size={10} /> Take Profit
                                </label>
                                <input
                                    type="number"
                                    value={tp}
                                    onChange={(e) => setTp(e.target.value)}
                                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-white font-mono focus:outline-none focus:border-emerald-500 transition-colors"
                                    placeholder="Price level"
                                />
                                {suggestedTP && (
                                    <button
                                        onClick={() => setTp(suggestedTP.toFixed(2))}
                                        className="text-[9px] text-emerald-500/70 hover:text-emerald-500 mt-1 font-bold uppercase"
                                    >
                                        Auto: ${suggestedTP.toFixed(2)}
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="bg-blue-500/5 border border-blue-500/10 rounded-2xl p-4 flex gap-3">
                        <Info className="text-blue-500 shrink-0" size={16} />
                        <p className="text-[11px] text-slate-400">
                            Stop-loss and take-profit are AI-recommended based on current volatility and key support/resistance levels.
                        </p>
                    </div>
                </div>

                {/* Footer */}
                <div className="p-6 bg-slate-900/50 border-t border-slate-800">
                    <button
                        onClick={() => onConfirm({
                            amount,
                            sl: sl ? parseFloat(sl) : undefined,
                            tp: tp ? parseFloat(tp) : undefined
                        })}
                        className={cn(
                            "w-full py-4 rounded-2xl font-black uppercase tracking-widest text-sm transition-all shadow-lg active:scale-95",
                            direction === 'buy'
                                ? "bg-emerald-500 hover:bg-emerald-400 text-emerald-950 shadow-emerald-500/20"
                                : "bg-rose-500 hover:bg-rose-400 text-rose-950 shadow-rose-500/20"
                        )}
                    >
                        Execute {direction} Order
                    </button>
                </div>
            </div>
        </div>
    );
}
