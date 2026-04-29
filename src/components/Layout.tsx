import React from 'react';
import { Link, useLocation } from 'wouter';
import { ThemeToggle } from './ThemeToggle';
import { Activity, LayoutDashboard, LineChart, History, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  const navItems = [
    { path: '/', label: 'Live Dashboard', icon: LayoutDashboard },
    { path: '/analytics', label: 'Detailed Analytics', icon: LineChart },
    { path: '/history', label: 'History Logs', icon: History },
    { path: '/about', label: 'About System', icon: Info },
  ];

  return (
    <div className="min-h-screen relative flex flex-col">
      {/* Background Image/Gradient */}
      <div 
        className="fixed inset-0 z-[-1] bg-cover bg-center bg-no-repeat transition-opacity duration-1000 dark:opacity-100 opacity-0"
        style={{ backgroundImage: `url(${import.meta.env.BASE_URL}images/dashboard-bg.png)` }}
      />
      {/* Light mode fallback bg */}
      <div className="fixed inset-0 z-[-2] bg-gradient-to-br from-blue-50 to-purple-50 dark:from-background dark:to-background transition-colors duration-500" />

      {/* Header */}
      <header className="sticky top-0 z-50 glass-panel border-b border-white/10 dark:border-white/5 shadow-sm rounded-none">
        <div className="max-w-[1600px] mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center shadow-lg">
              <Activity className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-display font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">
              Anamoly <span className="font-light">Watch</span>
            </h1>
          </div>

          <nav className="hidden md:flex items-center gap-1 bg-muted/50 p-1 rounded-full backdrop-blur-md border border-border">
            {navItems.map(item => {
              const active = location === item.path;
              return (
                <Link key={item.path} href={item.path} className="relative px-4 py-2 text-sm font-medium transition-colors">
                  <span className={cn("relative z-10 flex items-center gap-2", active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </span>
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 bg-primary rounded-full shadow-md"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-4">
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Mobile Nav (Bottom) */}
      <nav className="md:hidden fixed bottom-4 left-4 right-4 z-50 glass-panel rounded-2xl p-2 flex justify-around items-center">
        {navItems.map(item => (
          <Link key={item.path} href={item.path} className={cn(
            "p-3 rounded-xl flex flex-col items-center gap-1 transition-colors",
            location === item.path ? "bg-primary text-primary-foreground" : "text-muted-foreground"
          )}>
            <item.icon className="w-5 h-5" />
            <span className="text-[10px] font-medium">{item.label.split(' ')[0]}</span>
          </Link>
        ))}
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-[1600px] mx-auto p-4 md:p-6 lg:p-8 pb-24 md:pb-8">
        <motion.div
          key={location}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
          className="h-full"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
