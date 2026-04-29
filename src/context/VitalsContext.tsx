import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
import { VitalsData, computeSeverity, computeUrgencyScore, Severity } from '@/lib/vitalsLogic';

interface WatchColors {
  body: string;
  strap: string;
}

interface VitalsContextType {
  vitals: VitalsData;
  setVitals: React.Dispatch<React.SetStateAction<VitalsData>>;
  updateVital: (key: keyof VitalsData, value: any) => void;
  severity: Severity;
  urgency: number;
  watchColors: WatchColors;
  setWatchColors: React.Dispatch<React.SetStateAction<WatchColors>>;
  prediction: any;
  setPrediction: React.Dispatch<React.SetStateAction<any>>;
}

const defaultVitals: VitalsData = {
  bvp: 72,
  eda: 2.1,
  temp: 36.6,
  activity: 'resting',
  accel: 1.2
};

const defaultColors: WatchColors = {
  body: '#1e293b',
  strap: '#0ea5e9'
};

const VitalsContext = createContext<VitalsContextType | undefined>(undefined);

export function VitalsProvider({ children }: { children: React.ReactNode }) {
  const [vitals, setVitals] = useState<VitalsData>(defaultVitals);
  const [watchColors, setWatchColors] = useState<WatchColors>(defaultColors);
  const [prediction, setPrediction] = useState<any>(null);

  const updateVital = (key: keyof VitalsData, value: any) => {
    setVitals(prev => ({ ...prev, [key]: value }));
  };

  const severity = useMemo(() => computeSeverity(vitals), [vitals]);
  const urgency = useMemo(() => computeUrgencyScore(vitals), [vitals]);

  // 🔥 API CALL TO BACKEND
  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(vitals)
        });

        const data = await res.json();
        setPrediction(data);
      } catch (err) {
        console.error("API error:", err);
      }
    };

    fetchPrediction();
  }, [vitals]);

  return (
    <VitalsContext.Provider value={{ 
      vitals, 
      setVitals, 
      updateVital, 
      severity, 
      urgency,
      watchColors,
      setWatchColors,
      prediction,
      setPrediction
    }}>
      {children}
    </VitalsContext.Provider>
  );
}

export const useVitals = () => {
  const context = useContext(VitalsContext);
  if (!context) throw new Error('useVitals must be used within VitalsProvider');
  return context;
};