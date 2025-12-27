"use client"
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Signal, TrendingUp, Shield, BarChart3, Globe } from 'lucide-react';

export default function LandingPage() {
  const [mounted, setMounted] = useState(false);
  const [signals, setSignals] = useState<any[]>([]);

  useEffect(() => {
    setMounted(true);
    const fetchLatest = async () => {
      try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${API_URL}/signals`);
        const data = await res.json();
        setSignals(data.slice(0, 3));
      } catch (e) {
        console.error(e);
      }
    };
    fetchLatest();
  }, []);

  if (!mounted) return null;

  return (
    <main className="min-h-screen bg-[#020617] text-slate-200 overflow-x-hidden">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-4 md:px-8 max-w-7xl mx-auto">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-600/10 blur-[120px] rounded-full -z-10" />
        <div className="absolute bottom-0 left-0 w-[300px] h-[300px] bg-teal-600/10 blur-[100px] rounded-full -z-10" />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-bold uppercase tracking-wider">
              <Signal size={14} className="animate-pulse" /> Live Analysis Active
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold text-white tracking-tight leading-[1.1]">
              Decode the <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-400">Market</span> Noise.
            </h1>
            <p className="text-lg text-slate-400 max-w-xl leading-relaxed">
              Empower your trading with institutional-grade intelligence. Smark combines technical MACD analysis with AI sentiment scanning to give you the ultimate edge.
            </p>
            <div className="flex flex-wrap gap-4 pt-4">
              <Link href="/analysis" className="px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold flex items-center gap-2 shadow-lg shadow-blue-500/20 transition-all transform hover:-translate-y-1">
                Enter Terminal <ArrowRight size={20} />
              </Link>
              <button className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-bold transition-all border border-slate-700">
                View Performance
              </button>
            </div>
            {/* Quick Stats */}
            <div className="flex gap-12 pt-8">
              <div>
                <div className="text-3xl font-bold text-white">84.2%</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest mt-1">Accuracy</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white">2.4k+</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest mt-1">Pro Users</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white">24/7</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest mt-1">Scanning</div>
              </div>
            </div>
          </div>

          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-teal-600 rounded-3xl blur opacity-20 group-hover:opacity-30 transition duration-1000" />
            <div className="relative bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden shadow-2xl">
              <Image
                src="/hero.png"
                alt="Market Analysis Hero"
                width={800}
                height={600}
                className="w-full h-full object-cover transform transition duration-700 group-hover:scale-105"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Feature Snippets: Live Signals */}
      <section className="py-20 px-4 md:px-8 max-w-7xl mx-auto border-t border-slate-800/50">
        <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-8">
          <div className="space-y-4">
            <h2 className="text-3xl font-bold text-white">Live Edge Preview</h2>
            <p className="text-slate-500 max-w-md">Our algorithm never sleeps. Here are the most recent high-probability setups identified by Smark AI.</p>
          </div>
          <Link href="/analysis" className="text-blue-500 font-bold flex items-center gap-1 hover:underline">
            Visit Documentation <ArrowRight size={16} />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {signals.length > 0 ? signals.map((s, i) => (
            <div key={i} className="p-6 bg-slate-900/40 border border-slate-800 rounded-2xl hover:border-blue-500/30 transition-all group">
              <div className="flex justify-between items-start mb-4">
                <span className="font-bold text-white group-hover:text-blue-400 transition-colors uppercase tracking-wider">{s.ticker}</span>
                <span className="text-[10px] bg-blue-500/10 text-blue-400 px-2 py-1 rounded-md border border-blue-500/20 font-bold uppercase">{s.type}</span>
              </div>
              <div className="flex justify-between items-end">
                <div className="space-y-1">
                  <p className="text-[10px] text-slate-500 uppercase font-black tracking-tighter">Confidence</p>
                  <p className="text-2xl font-mono text-white leading-none">{s.confidence}%</p>
                </div>
                <div className="p-3 bg-blue-600/10 rounded-xl">
                  <TrendingUp size={24} className="text-blue-500" />
                </div>
              </div>
            </div>
          )) : (
            [1, 2, 3].map(i => (
              <div key={i} className="p-6 bg-slate-900/40 border border-slate-800 rounded-2xl animate-pulse h-32" />
            ))
          )}
        </div>
      </section>

      {/* Core Values */}
      <section className="py-20 px-4 md:px-8 bg-slate-900/20">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-12">
          <div className="space-y-4">
            <div className="w-12 h-12 bg-blue-600/10 rounded-2xl flex items-center justify-center text-blue-500 border border-blue-500/20">
              <Shield size={24} />
            </div>
            <h3 className="text-xl font-bold text-white">Risk Intelligence</h3>
            <p className="text-slate-500 text-sm leading-relaxed">Integrated risk management filters every signal, ensuring your capital is always protected by calculated probability.</p>
          </div>
          <div className="space-y-4">
            <div className="w-12 h-12 bg-teal-600/10 rounded-2xl flex items-center justify-center text-teal-500 border border-teal-500/20">
              <BarChart3 size={24} />
            </div>
            <h3 className="text-xl font-bold text-white">Pattern Recognition</h3>
            <p className="text-slate-500 text-sm leading-relaxed">From MACD Crossovers to RSI Divergence, we utilize battle-tested technical frameworks optimized for modern markets.</p>
          </div>
          <div className="space-y-4">
            <div className="w-12 h-12 bg-indigo-600/10 rounded-2xl flex items-center justify-center text-indigo-500 border border-indigo-500/20">
              <Globe size={24} />
            </div>
            <h3 className="text-xl font-bold text-white">Sentiment Alpha</h3>
            <p className="text-slate-500 text-sm leading-relaxed">Don't just follow price. Our AI scans global financial news in real-time to detect the sentiment shifts before they hit the chart.</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-slate-800 text-center">
        <p className="text-slate-600 text-xs font-bold uppercase tracking-[0.2em]">Smark Terminal © 2025 • Institutional Grade Alpha</p>
      </footer>
    </main>
  );
}
