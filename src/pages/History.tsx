import React, { useMemo } from 'react';
import { useVitals } from '@/context/VitalsContext';
import { computeSeverity } from '@/lib/vitalsLogic';
import { Clock, Download, Filter } from 'lucide-react';
import { cn } from '@/lib/utils';

export function History() {
  const { vitals } = useVitals();

  // Generate fake history log table based on current state + random noise
  const logs = useMemo(() => {
    const data = [];
    let currentV = { ...vitals };
    const now = new Date();
    
    for(let i=0; i<20; i++) {
      const time = new Date(now.getTime() - i * 5 * 60000); // 5 min intervals
      const entry = {
        id: `LOG-${1000 - i}`,
        timestamp: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        date: time.toLocaleDateString(),
        vitals: { ...currentV },
        severity: computeSeverity(currentV)
      };
      data.push(entry);

      // Walk values slightly
      currentV.bvp += (Math.random() * 10 - 5);
      currentV.eda += (Math.random() * 2 - 1);
      currentV.temp += (Math.random() * 0.4 - 0.2);
    }
    return data;
  }, [vitals]);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold">History Logs</h2>
          <p className="text-muted-foreground mt-2">
            Chronological record of your vital signs and automated system assessments.
          </p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 rounded-xl bg-card border border-border text-sm font-medium flex items-center gap-2 hover:bg-muted transition-colors">
            <Filter className="w-4 h-4" /> Filter
          </button>
          <button className="px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium flex items-center gap-2 shadow-lg shadow-primary/20 hover:-translate-y-0.5 transition-transform">
            <Download className="w-4 h-4" /> Export CSV
          </button>
        </div>
      </header>

      <div className="glass-panel rounded-3xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="p-4 font-semibold text-sm text-muted-foreground">Time</th>
                <th className="p-4 font-semibold text-sm text-muted-foreground">BVP (bpm)</th>
                <th className="p-4 font-semibold text-sm text-muted-foreground">EDA (µS)</th>
                <th className="p-4 font-semibold text-sm text-muted-foreground">Temp (°C)</th>
                <th className="p-4 font-semibold text-sm text-muted-foreground">Activity</th>
                <th className="p-4 font-semibold text-sm text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, i) => (
                <tr key={log.id} className={cn(
                  "border-b border-border/50 hover:bg-muted/20 transition-colors",
                  i === 0 && "bg-primary/5 hover:bg-primary/10" // Highlight newest
                )}>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      {i === 0 && <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />}
                      <div>
                        <div className="font-medium">{log.timestamp}</div>
                        <div className="text-xs text-muted-foreground">{log.date}</div>
                      </div>
                    </div>
                  </td>
                  <td className="p-4 font-mono font-medium">{log.vitals.bvp.toFixed(0)}</td>
                  <td className="p-4 font-mono font-medium">{log.vitals.eda.toFixed(1)}</td>
                  <td className="p-4 font-mono font-medium">{log.vitals.temp.toFixed(1)}</td>
                  <td className="p-4 capitalize">{log.vitals.activity}</td>
                  <td className="p-4">
                    <span className={cn(
                      "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider inline-block",
                      log.severity === 'critical' ? 'bg-red-500/20 text-red-500 border border-red-500/30' :
                      log.severity === 'high' ? 'bg-orange-500/20 text-orange-500 border border-orange-500/30' :
                      log.severity === 'moderate' ? 'bg-yellow-500/20 text-yellow-600 border border-yellow-500/30' :
                      'bg-emerald-500/20 text-emerald-500 border border-emerald-500/30'
                    )}>
                      {log.severity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
