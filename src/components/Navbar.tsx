"use client"
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, LayoutDashboard, Home as HomeIcon } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export function Navbar() {
    const pathname = usePathname();

    const navItems = [
        { name: 'Home', href: '/', icon: HomeIcon },
        { name: 'Analysis', href: '/analysis', icon: LayoutDashboard },
    ];

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 flex justify-center p-4">
            <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-full px-6 py-2 flex items-center gap-8 shadow-2xl">
                <Link href="/" className="flex items-center gap-2 mr-4">
                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <Activity className="text-white" size={18} />
                    </div>
                    <span className="text-white font-bold tracking-tighter">SMARK</span>
                </Link>
                <div className="flex items-center gap-1">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200",
                                    isActive
                                        ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
                                        : "text-slate-400 hover:text-white hover:bg-slate-800"
                                )}
                            >
                                <item.icon size={16} />
                                {item.name}
                            </Link>
                        );
                    })}
                </div>
            </div>
        </nav>
    );
}
