import React from 'react';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface SignalProps {
  signal: {
    ticker: string;
    type: string;
    confidence: number;
    entry: number;
    sl: number;
    tp: number;
    created_at: string;
  };
}

export const SignalCard: React.FC<SignalProps> = ({ signal }) => {
  const isBullish = signal.type.toLowerCase().includes('bullish');
  const statusColor = signal.confidence > 80 ? 'text-emerald-400' : 'text-amber-400';

  return (
    <div className="bg-slate-900/50 backdrop-blur-md p-4 rounded-xl border border-slate-800 mb-4 hover:border-blue-500/50 transition-all group shadow-lg">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <div className={cn(
            "p-1.5 rounded-lg",
            isBullish ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"
          )}>
            {isBullish ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
          </div>
          <span className="font-bold text-white text-lg tracking-tight">{signal.ticker}</span>
        </div>
        <div className={cn(
          "text-[10px] font-bold px-2 py-1 rounded-full bg-slate-800/80 border border-slate-700/50 uppercase tracking-wider",
          statusColor
        )}>
          {signal.confidence}% Confidence
        </div>
      </div>
      
      <p className="text-slate-400 text-sm mb-4 font-medium">
        {signal.type.replace(/_/g, ' ')}
      </p>

      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-800/40 p-2 rounded-lg border border-slate-700/30">
          <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">Entry</p>
          <p className="text-white font-mono text-sm">${signal.entry.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/40 p-2 rounded-lg border border-slate-700/30">
          <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">Stop Loss</p>
          <p className="text-rose-400 font-mono text-sm">${signal.sl.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/40 p-2 rounded-lg border border-slate-700/30">
          <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">Take Profit</p>
          <p className="text-emerald-400 font-mono text-sm">${signal.tp.toFixed(2)}</p>
        </div>
      </div>
      
      <div className="mt-3 pt-3 border-t border-slate-800 flex justify-between items-center">
        <span className="text-[10px] text-slate-500 font-mono">
          {new Date(signal.created_at).toLocaleTimeString()}
        </span>
        <button className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors flex items-center gap-1 font-bold uppercase tracking-wider">
          View Details <AlertCircle size={10} />
        </button>
      </div>
    </div>
  );
};
