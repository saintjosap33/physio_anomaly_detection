import React from 'react';
import { useVitals } from '@/context/VitalsContext';
import { Heart, Activity, Thermometer, Footprints, Settings2, Zap } from 'lucide-react';

export function VitalsInput() {
  const { vitals, updateVital, watchColors, setWatchColors } = useVitals();

  // Safe values (prevents crash if undefined)
  const safeVitals = {
    bvp: vitals.bvp ?? 75,
    eda: vitals.eda ?? 2,
    temp: vitals.temp ?? 36.6,
    accel: vitals.accel ?? 0,
    activity: vitals.activity ?? 'resting'
  };

  return (
    <div className="glass-panel rounded-3xl p-6 flex flex-col gap-6 h-full">
      
      {/* HEADER */}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-1">
          <Settings2 className="text-primary" />
          Manual Vitals
        </h2>
        <p className="text-sm text-muted-foreground">
          Adjust sliders to simulate sensor readings
        </p>
      </div>

      {/* INPUTS */}
      <div className="space-y-6">

        {/* BVP */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-semibold flex items-center gap-2">
              <Heart className="w-4 h-4 text-red-500" />
              Heart Rate (BVP)
            </label>
            <span className="font-mono font-bold text-lg">
              {safeVitals.bvp} <span className="text-xs text-muted-foreground">bpm</span>
            </span>
          </div>

          <input
            type="range"
            min="30"
            max="200"
            value={safeVitals.bvp}
            onChange={(e) => updateVital('bvp', Number(e.target.value))}
            className="w-full accent-red-500"
          />

          <div className="flex justify-between text-[10px] text-muted-foreground uppercase font-bold">
            <span>Brady</span>
            <span>Normal</span>
            <span>Tachy</span>
          </div>
        </div>

        {/* EDA */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-semibold flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-500" />
              Stress (EDA)
            </label>
            <span className="font-mono font-bold text-lg">
              {safeVitals.eda.toFixed(1)} <span className="text-xs text-muted-foreground">µS</span>
            </span>
          </div>

          <input
            type="range"
            min="0"
            max="20"
            step="0.1"
            value={safeVitals.eda}
            onChange={(e) => updateVital('eda', Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
        </div>

        {/* TEMP */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-semibold flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-orange-500" />
              Temperature
            </label>
            <span className="font-mono font-bold text-lg">
              {safeVitals.temp.toFixed(1)} °C
            </span>
          </div>

          <input
            type="range"
            min="35"
            max="42"
            step="0.1"
            value={safeVitals.temp}
            onChange={(e) => updateVital('temp', Number(e.target.value))}
            className="w-full accent-orange-500"
          />
        </div>

        {/* ACCEL */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-semibold flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-500" />
              Accelerometer
            </label>
            <span className="font-mono font-bold text-lg">
              {safeVitals.accel.toFixed(1)} g
            </span>
          </div>

          <input
            type="range"
            min="0"
            max="10"
            step="0.1"
            value={safeVitals.accel}
            onChange={(e) => updateVital('accel', Number(e.target.value))}
            className="w-full accent-purple-500"
          />
        </div>

        {/* ACTIVITY */}
        <div className="space-y-2">
          <label className="text-sm font-semibold flex items-center gap-2">
            <Footprints className="w-4 h-4 text-purple-500" />
            Activity Level
          </label>

          <select
            value={safeVitals.activity}
            onChange={(e) => updateVital('activity', e.target.value as any)}
            className="w-full bg-background border border-border rounded-xl p-3 text-sm cursor-pointer"
          >
            <option value="sleeping">Sleeping</option>
            <option value="resting">Resting</option>
            <option value="working">Working</option>
            <option value="walking">Walking</option>
            <option value="running">Running</option>
          </select>
        </div>
      </div>

      {/* WATCH CUSTOMIZATION */}
      <div className="mt-auto pt-6 border-t border-border">
        <h3 className="text-sm font-semibold mb-3">Watch Customization</h3>

        <div className="grid grid-cols-2 gap-4">

          {/* STRAP */}
          <div>
            <label className="text-xs text-muted-foreground mb-2 block">
              Strap Color
            </label>

            <div className="flex gap-2">
              {['#0ea5e9', '#ef4444', '#10b981', '#f97316', '#a855f7'].map(c => (
                <button
                  key={c}
                  onClick={() => setWatchColors(prev => ({ ...prev, strap: c }))}
                  className={`w-6 h-6 rounded-full border-2 ${
                    watchColors.strap === c ? 'border-white' : 'border-transparent'
                  }`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>

          {/* BODY */}
          <div>
            <label className="text-xs text-muted-foreground mb-2 block">
              Body Color
            </label>

            <div className="flex gap-2">
              {['#1e293b', '#e2e8f0', '#b45309', '#0f172a'].map(c => (
                <button
                  key={c}
                  onClick={() => setWatchColors(prev => ({ ...prev, body: c }))}
                  className={`w-6 h-6 rounded-full border-2 ${
                    watchColors.body === c ? 'border-white' : 'border-transparent'
                  }`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}