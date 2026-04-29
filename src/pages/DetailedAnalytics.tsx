import React, { useMemo } from 'react';
import { useVitals } from '@/context/VitalsContext';
import { generateHistorySeries } from '@/lib/vitalsLogic';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, Legend
} from 'recharts';
import { Activity, Heart, Thermometer } from 'lucide-react';

export function DetailedAnalytics() {
  const { vitals } = useVitals();

  // Generate larger datasets for full page
  const hrData = useMemo(() => generateHistorySeries(vitals.bvp, 60, 8).map(d => ({...d, baseline: 75})), [vitals.bvp]);
  const edaData = useMemo(() => generateHistorySeries(vitals.eda, 60, 1.5).map(d => ({...d, baseline: 2.0})), [vitals.eda]);
  const tempData = useMemo(() => generateHistorySeries(vitals.temp, 60, 0.2).map(d => ({...d, baseline: 36.6})), [vitals.temp]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-panel p-3 rounded-xl border border-border">
          <p className="font-semibold text-sm mb-1">{label}</p>
          {payload.map((p: any, idx: number) => (
             <p key={idx} className="text-sm font-mono flex items-center gap-2" style={{ color: p.color }}>
               <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
               {p.name}: {p.value.toFixed(1)}
             </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h2 className="text-3xl font-display font-bold">Detailed Analytics</h2>
        <p className="text-muted-foreground mt-2 max-w-2xl">
          Comprehensive view of your vital trends over the last 60 minutes. Baseline comparisons indicate deviations from your personal healthy averages.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Heart Rate Chart */}
        <div className="glass-panel rounded-3xl p-6">
          <h3 className="text-xl font-bold flex items-center gap-2 mb-6">
            <Heart className="text-red-500" /> Heart Rate Trend
          </h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer>
              <AreaChart data={hrData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} vertical={false} />
                <XAxis dataKey="time" stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} domain={['dataMin - 10', 'dataMax + 10']} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
                <Area type="monotone" dataKey="value" name="Current BVP" stroke="#ef4444" strokeWidth={3} fill="url(#hrGrad)" isAnimationActive={false} />
                <Area type="step" dataKey="baseline" name="Baseline" stroke="currentColor" strokeOpacity={0.3} strokeDasharray="5 5" fill="none" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* EDA Chart */}
        <div className="glass-panel rounded-3xl p-6">
          <h3 className="text-xl font-bold flex items-center gap-2 mb-6">
            <Activity className="text-cyan-500" /> Electrodermal Activity
          </h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer>
              <BarChart data={edaData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} vertical={false} />
                <XAxis dataKey="time" stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'currentColor', opacity: 0.05 }} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
                <Bar dataKey="value" name="EDA (µS)" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Area type="step" dataKey="baseline" name="Baseline" stroke="currentColor" strokeOpacity={0.3} strokeDasharray="5 5" fill="none" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Temperature Chart */}
        <div className="glass-panel rounded-3xl p-6 lg:col-span-2">
          <h3 className="text-xl font-bold flex items-center gap-2 mb-6">
            <Thermometer className="text-orange-500" /> Skin Temperature
          </h3>
          <div className="h-[250px] w-full">
            <ResponsiveContainer>
              <AreaChart data={tempData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} vertical={false} />
                <XAxis dataKey="time" stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="currentColor" opacity={0.5} fontSize={12} tickLine={false} axisLine={false} domain={[34, 42]} />
                <Tooltip content={<CustomTooltip />} />
                 <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
                <Area type="monotone" dataKey="value" name="Temp °C" stroke="#f97316" strokeWidth={3} fill="url(#tempGrad)" isAnimationActive={false} />
                 <Area type="step" dataKey="baseline" name="Baseline" stroke="currentColor" strokeOpacity={0.3} strokeDasharray="5 5" fill="none" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
