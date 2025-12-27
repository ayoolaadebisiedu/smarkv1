"use client"
import React from 'react';
import { History, TrendingUp, TrendingDown, Clock, ArrowRight } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface TradeHistoryProps {
    trades: any[];
}

export function TradeHistory({ trades }: TradeHistoryProps) {
    return (
        <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 shadow-xl h-full flex flex-col">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2 uppercase tracking-tighter shrink-0">
                Performance Log <History size={20} className="text-blue-500" />
            </h2>

            <div className="flex-1 overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-slate-800">
                {trades.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-20 grayscale opacity-30 text-center">
                        <History size={48} className="text-slate-600 mb-4" />
                        <p className="text-xs font-bold uppercase tracking-widest">No completed trades</p>
                    </div>
                ) : (
                    trades.map((trade) => (
                        <div key={trade.id} className="p-4 bg-slate-950/40 border border-slate-800/50 rounded-2xl hover:border-slate-700 transition-colors group">
                            <div className="flex justify-between items-start mb-3">
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-bold text-white uppercase text-sm">{trade.ticker}</span>
                                        <span className={cn(
                                            "text-[9px] px-2 py-0.5 rounded-md font-black uppercase",
                                            trade.direction === 'buy' ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"
                                        )}>
                                            {trade.direction === 'buy' ? 'Long' : 'Short'}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                                        <Clock size={10} />
                                        <span>{new Date(trade.closed_at).toLocaleDateString()} {new Date(trade.closed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <span className={cn(
                                        "text-sm font-black font-mono",
                                        trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                    )}>
                                        {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                                    </span>
                                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-tighter">Realized P&L</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-4 pt-3 border-t border-slate-800/50">
                                <div className="space-y-0.5">
                                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block">Entry</span>
                                    <span className="text-xs font-mono text-slate-300">${trade.entry_price.toFixed(2)}</span>
                                </div>
                                <ArrowRight size={14} className="text-slate-700" />
                                <div className="space-y-0.5">
                                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block">Exit</span>
                                    <span className="text-xs font-mono text-slate-300">${trade.exit_price.toFixed(2)}</span>
                                </div>
                                <div className="ml-auto space-y-0.5">
                                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block">Amount</span>
                                    <span className="text-xs font-mono text-slate-300">{trade.amount} units</span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
