export interface VitalsData {
  bvp: number; // Blood Volume Pulse (bpm)
  eda: number; // Electrodermal Activity (µS)
  temp: number; // Skin Temperature (°C)
  activity: 'resting' | 'walking' | 'running' | 'working' | 'sleeping';
  accel: number; // Accelerometer intensity 0-10
}

export type Severity = 'low' | 'moderate' | 'high' | 'critical';


// ===============================
// 🔥 BACKEND API CALL (NEW)
// ===============================
export async function sendVitals(data: VitalsData) {
  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        bvp: data.bvp,
        eda: data.eda,
        temp: data.temp,
        accel: data.accel,
      }),
    });

    const result = await res.json();
    return result;

  } catch (error) {
    console.error("Backend error:", error);
    return { status: "error", message: "Backend not reachable" };
  }
}


// ===============================
// EXISTING LOGIC (UNCHANGED)
// ===============================
export function computeSeverity(vitals: VitalsData): Severity {
  const { bvp, eda, temp } = vitals;
  let maxSev = 0;

  if (bvp < 50 || bvp > 160) maxSev = Math.max(maxSev, 3);
  else if ((bvp >= 50 && bvp < 60) || (bvp > 120 && bvp <= 160)) maxSev = Math.max(maxSev, 2);
  else if (bvp >= 100 && bvp <= 120) maxSev = Math.max(maxSev, 1);

  if (eda > 15) maxSev = Math.max(maxSev, 3);
  else if (eda > 10) maxSev = Math.max(maxSev, 2);
  else if (eda > 5) maxSev = Math.max(maxSev, 1);

  if (temp > 39.5 || temp < 35.0) maxSev = Math.max(maxSev, 3);
  else if (temp > 38.5) maxSev = Math.max(maxSev, 2);
  else if (temp > 37.5) maxSev = Math.max(maxSev, 1);

  const severities: Severity[] = ['low', 'moderate', 'high', 'critical'];
  return severities[maxSev];
}

export function computeUrgencyScore(vitals: VitalsData): number {
  const { bvp, eda, temp } = vitals;
  let score = 0;
  
  const bvpDeviation = Math.abs(bvp - 75) / 75;
  score += Math.min(bvpDeviation * 40, 40);
  
  const edaDeviation = Math.abs(eda - 2) / 2;
  score += Math.min(edaDeviation * 30, 30);
  
  const tempDeviation = Math.abs(temp - 36.6);
  score += Math.min(tempDeviation * 15, 30);
  
  return Math.min(Math.round(score), 100);
}

export function detectAnomalies(vitals: VitalsData): string[] {
  const anomalies: string[] = [];

  if (vitals.bvp > 120 && vitals.activity === 'resting') {
    anomalies.push('Tachycardia at rest');
  }
  if (vitals.bvp < 50) anomalies.push('Bradycardia detected');
  if (vitals.eda > 10) anomalies.push('Abnormal stress response (High EDA)');
  if (vitals.temp > 38.0) anomalies.push('Elevated body temperature');

  return anomalies;
}

export function predictConditions(vitals: VitalsData) {
  let stress = Math.min(Math.round((vitals.eda / 15) * 100), 100);
  let fatigue = 10;
  
  if (vitals.activity === 'working' && vitals.eda > 5) fatigue += 30;
  if (vitals.bvp > 90 && vitals.activity === 'resting') fatigue += 20;
  
  let feverRisk = Math.min(Math.round(((vitals.temp - 36.5) / 3) * 100), 100);
  feverRisk = Math.max(feverRisk, 0);
  
  let irregularHR = 5;
  if (vitals.bvp > 100 || vitals.bvp < 55) irregularHR = 65;

  return {
    stress: Math.min(stress, 100),
    fatigue: Math.min(fatigue, 100),
    feverRisk: Math.min(feverRisk, 100),
    irregularHR: Math.min(irregularHR, 100)
  };
}

export function generateHistorySeries(currentValue: number, count: number, variance: number) {
  const data = [];
  let val = currentValue;

  for (let i = 0; i < count; i++) {
    val = val + (Math.random() * variance * 2 - variance);
    data.unshift(Number(val.toFixed(1)));
  }

  data.push(currentValue);

  return data.map((v, i) => ({
    time: `-${count - i}m`,
    value: v
  }));
}