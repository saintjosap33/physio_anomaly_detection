import React, { useMemo } from 'react';
import { useVitals } from '@/context/VitalsContext';
import { 
  detectAnomalies, 
  predictConditions, 
  generateHistorySeries 
} from '@/lib/vitalsLogic';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { AlertTriangle, Brain, ShieldAlert, CheckCircle2, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export function AnalyticsPanel() {
  const { vitals, severity, urgency } = useVitals();

  const anomalies = useMemo(() => detectAnomalies(vitals), [vitals]);
  const predictions = useMemo(() => predictConditions(vitals), [vitals]);
  
  // Generate some fake history based on current value to make charts look alive
  const hrData = useMemo(() => generateHistorySeries(vitals.bvp, 10, 5), [vitals.bvp]);
  const edaData = useMemo(() => generateHistorySeries(vitals.eda, 10, 1), [vitals.eda]);

  const severityColors = {
    low: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
    moderate: 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20',
    high: 'text-orange-500 bg-orange-500/10 border-orange-500/20',
    critical: 'text-red-500 bg-red-500/10 border-red-500/20 neon-glow-destructive',
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Top Status Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className={cn(
          "glass-panel rounded-2xl p-4 flex flex-col justify-center items-center text-center transition-all duration-300",
          severityColors[severity]
        )}>
          <ShieldAlert className="w-8 h-8 mb-2" />
          <h3 className="text-sm font-semibold uppercase tracking-wider opacity-80">System Status</h3>
          <div className="text-2xl font-bold font-display capitalize">{severity}</div>
        </div>

        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-center items-center text-center">
          <Zap className="w-8 h-8 mb-2 text-primary" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Urgency Score</h3>
          <div className="text-3xl font-bold font-display text-primary">{urgency} <span className="text-sm text-muted-foreground font-sans font-normal">/100</span></div>
        </div>
      </div>

      {/* AI Predictions */}
      <div className="glass-panel rounded-2xl p-5 flex-1">
        <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
          <Brain className="text-secondary" />
          AI Health Predictions
        </h3>
        
        <div className="space-y-4">
          <PredictionBar label="Stress Level" value={predictions.stress} color="bg-cyan-500" />
          <PredictionBar label="Fatigue Risk" value={predictions.fatigue} color="bg-purple-500" />
          <PredictionBar label="Fever Risk" value={predictions.feverRisk} color="bg-orange-500" />
          <PredictionBar label="Irregular HR" value={predictions.irregularHR} color="bg-red-500" />
        </div>
      </div>

      {/* Anomalies List */}
      <div className="glass-panel rounded-2xl p-5 flex-1 flex flex-col">
        <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
          <AlertTriangle className="text-yellow-500" />
          Live Anomalies
        </h3>
        
        <div className="flex-1 overflow-y-auto pr-2">
          {anomalies.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-70">
              <CheckCircle2 className="w-10 h-10 mb-2 text-emerald-500" />
              <p>No anomalies detected</p>
            </div>
          ) : (
            <ul className="space-y-3">
              {anomalies.map((anomaly, idx) => (
                <motion.li 
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  key={idx} 
                  className="bg-destructive/10 border border-destructive/20 text-destructive rounded-xl p-3 text-sm flex items-start gap-3"
                >
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{anomaly}</span>
                </motion.li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Mini Sparkline Chart */}
      <div className="glass-panel rounded-2xl p-5 h-40 flex flex-col">
        <h3 className="text-sm font-semibold mb-2 text-muted-foreground">BVP Trend (Live)</h3>
        <div className="flex-1 -mx-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={hrData}>
              <defs>
                <linearGradient id="colorBvp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorBvp)" 
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function PredictionBar({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1 font-medium">
        <span>{label}</span>
        <span className="text-muted-foreground">{value}%</span>
      </div>
      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
        <motion.div 
          className={`h-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
