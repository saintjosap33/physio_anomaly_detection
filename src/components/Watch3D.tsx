import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useVitals } from '@/context/VitalsContext';
import { Heart, Activity, Thermometer, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

const STRAP_COLORS = ['#1e1e2e', '#4a4a6a', '#1a3a5c', '#1a4a2e', '#4a1a1a', '#6b21a8'];
const BODY_COLORS = ['#d4d4d8', '#a1a1aa', '#cbd5e1', '#c7d2fe', '#fde68a', '#fcd5ce'];

function ECGWave({ bvp }: { bvp: number }) {
  const period = Math.max(600, Math.round(60000 / bvp));
  return (
    <svg viewBox="0 0 200 40" className="w-full h-8" preserveAspectRatio="none">
      <defs>
        <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0" />
          <stop offset="30%" stopColor="#10b981" stopOpacity="1" />
          <stop offset="70%" stopColor="#06b6d4" stopOpacity="1" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
        </linearGradient>
      </defs>
      <motion.path
        d="M0,20 L30,20 L35,20 L40,5 L45,35 L50,20 L60,20 L62,18 L65,10 L68,22 L72,20 L80,20 L100,20 L130,20 L135,20 L140,5 L145,35 L150,20 L160,20 L162,18 L165,10 L168,22 L172,20 L200,20"
        fill="none"
        stroke="url(#ecgGrad)"
        strokeWidth="1.5"
        strokeLinecap="round"
        animate={{ x: [0, -100] }}
        transition={{ repeat: Infinity, duration: period / 1000, ease: 'linear' }}
      />
    </svg>
  );
}

function PPGWave({ bvp }: { bvp: number }) {
  const period = Math.max(600, Math.round(60000 / bvp));
  const points = Array.from({ length: 20 }, (_, i) => {
    const x = (i / 19) * 200;
    const y = 20 - Math.sin((i / 19) * Math.PI * 2) * 12;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg viewBox="0 0 200 40" className="w-full h-6" preserveAspectRatio="none">
      <defs>
        <linearGradient id="ppgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#f43f5e" stopOpacity="0" />
          <stop offset="50%" stopColor="#f43f5e" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#f43f5e" stopOpacity="0" />
        </linearGradient>
      </defs>
      <motion.polyline
        points={points}
        fill="none"
        stroke="url(#ppgGrad)"
        strokeWidth="1.5"
        strokeLinecap="round"
        animate={{ x: [0, -100] }}
        transition={{ repeat: Infinity, duration: period / 1000, ease: 'linear' }}
      />
    </svg>
  );
}

export function Watch3D() {
  const { vitals, severity, watchColors, setWatchColors } = useVitals();

  const getSeverityColor = () => {
    switch (severity) {
      case 'critical': return '#ef4444';
      case 'high': return '#f97316';
      case 'moderate': return '#eab308';
      default: return '#10b981';
    }
  };

  const getSeverityGlow = () => {
    switch (severity) {
      case 'critical': return 'shadow-[0_0_40px_rgba(239,68,68,0.5)]';
      case 'high': return 'shadow-[0_0_40px_rgba(249,115,22,0.4)]';
      case 'moderate': return 'shadow-[0_0_40px_rgba(234,179,8,0.3)]';
      default: return 'shadow-[0_0_40px_rgba(16,185,129,0.2)]';
    }
  };

  const heartPeriod = Math.max(0.3, 60 / vitals.bvp);

  return (
    <div className="w-full flex flex-col items-center gap-4">
      {/* Watch container */}
      <div className="relative flex items-center justify-center" style={{ width: 220, height: 380 }}>
        {/* Top strap */}
        <div
          className="absolute rounded-t-xl"
          style={{
            width: 100,
            height: 110,
            top: 0,
            backgroundColor: watchColors.strap,
            zIndex: 1,
          }}
        />
        {/* Bottom strap */}
        <div
          className="absolute rounded-b-xl"
          style={{
            width: 100,
            height: 110,
            bottom: 0,
            backgroundColor: watchColors.strap,
            zIndex: 1,
          }}
        />

        {/* Health risk ring */}
        <motion.div
          className="absolute rounded-[2.5rem] z-10 pointer-events-none"
          style={{
            width: 170,
            height: 200,
            border: `3px solid ${getSeverityColor()}`,
            boxShadow: `0 0 20px ${getSeverityColor()}88, inset 0 0 10px ${getSeverityColor()}22`,
            top: '50%',
            left: '50%',
            marginLeft: -85,
            marginTop: -100,
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
        />

        {/* Watch body */}
        <div
          className={cn(
            'absolute rounded-[2.2rem] z-20 flex items-center justify-center',
            getSeverityGlow()
          )}
          style={{
            width: 160,
            height: 190,
            top: '50%',
            left: '50%',
            marginLeft: -80,
            marginTop: -95,
            backgroundColor: watchColors.body,
            boxShadow: `0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2)`,
          }}
        >
          {/* Watch screen */}
          <div
            className="rounded-[1.8rem] flex flex-col overflow-hidden"
            style={{
              width: 140,
              height: 170,
              background: 'linear-gradient(145deg, #0a0a12 0%, #12121f 100%)',
              boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.8)',
            }}
          >
            {/* Screen header */}
            <div className="flex justify-between items-center px-3 pt-2 pb-1">
              <span className="text-[9px] text-gray-400 font-medium">
                {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
              <span
                className="text-[8px] px-1.5 py-0.5 rounded-full font-bold"
                style={{
                  color: getSeverityColor(),
                  backgroundColor: `${getSeverityColor()}22`,
                }}
              >
                {severity.toUpperCase()}
              </span>
            </div>

            {/* BVP / Heart */}
            <div className="flex items-center gap-2 px-3 py-1">
              <motion.div
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ repeat: Infinity, duration: heartPeriod, ease: 'easeInOut' }}
              >
                <Heart className="w-5 h-5 text-red-500 fill-red-500" />
              </motion.div>
              <div>
                <span className="text-white font-bold text-2xl leading-none">{vitals.bvp}</span>
                <span className="text-gray-400 text-[9px] ml-1">BPM</span>
              </div>
            </div>

            {/* Other vitals */}
            <div className="grid grid-cols-2 gap-1 px-2 pb-1">
              <div className="rounded-lg p-1.5" style={{ background: 'rgba(6,182,212,0.1)' }}>
                <Activity className="w-3 h-3 text-cyan-400 mb-0.5" />
                <div className="text-white text-xs font-bold">{vitals.eda.toFixed(1)}</div>
                <div className="text-gray-500 text-[8px]">µS EDA</div>
              </div>
              <div className="rounded-lg p-1.5" style={{ background: 'rgba(249,115,22,0.1)' }}>
                <Thermometer className="w-3 h-3 text-orange-400 mb-0.5" />
                <div className="text-white text-xs font-bold">{vitals.temp.toFixed(1)}</div>
                <div className="text-gray-500 text-[8px]">°C TMP</div>
              </div>
            </div>

            {/* Accelerometer */}
            <div className="flex items-center gap-1.5 px-2 pb-1">
              <Zap className="w-3 h-3 text-purple-400 shrink-0" />
              <div className="flex-1 h-1.5 rounded-full bg-gray-700 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: 'linear-gradient(to right, #a855f7, #06b6d4)' }}
                  animate={{ width: `${(vitals.accel / 10) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <span className="text-[8px] text-gray-400">{vitals.accel.toFixed(1)}</span>
            </div>

            {/* PPG wave */}
            <div className="px-2 overflow-hidden">
              <PPGWave bvp={vitals.bvp} />
            </div>

            {/* ECG wave */}
            <div className="px-2 overflow-hidden mt-auto">
              <ECGWave bvp={vitals.bvp} />
            </div>
          </div>
        </div>

        {/* Crown button */}
        <div
          className="absolute rounded-full z-30"
          style={{
            width: 8,
            height: 32,
            right: 16,
            top: '50%',
            marginTop: -50,
            backgroundColor: watchColors.body,
            boxShadow: '2px 0 8px rgba(0,0,0,0.4)',
          }}
        />
      </div>

      {/* Color swatches */}
      <div className="flex flex-col gap-2 w-full max-w-xs">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">Strap</span>
          <div className="flex gap-1.5 flex-wrap">
            {STRAP_COLORS.map((color) => (
              <button
                key={color}
                onClick={() => setWatchColors((c) => ({ ...c, strap: color }))}
                className={cn(
                  'w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 focus:outline-none',
                  watchColors.strap === color ? 'border-white scale-110' : 'border-transparent'
                )}
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14 shrink-0">Case</span>
          <div className="flex gap-1.5 flex-wrap">
            {BODY_COLORS.map((color) => (
              <button
                key={color}
                onClick={() => setWatchColors((c) => ({ ...c, body: color }))}
                className={cn(
                  'w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 focus:outline-none',
                  watchColors.body === color ? 'border-primary scale-110' : 'border-transparent'
                )}
                style={{ backgroundColor: color, boxShadow: '0 1px 4px rgba(0,0,0,0.3)' }}
              />
            ))}
          </div>
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground flex items-center gap-1.5">
        <span
          className="w-1.5 h-1.5 rounded-full animate-pulse inline-block"
          style={{ backgroundColor: getSeverityColor() }}
        />
        Live vitals display · Color-customizable
      </p>
    </div>
  );
}
