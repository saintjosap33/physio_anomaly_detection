import React from 'react';
import { Shield, Brain, Watch, Cpu } from 'lucide-react';

export function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-8 py-8">
      <div className="text-center space-y-4 mb-12">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center shadow-2xl shadow-primary/30">
          <Watch className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight">Aegis Health System</h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Next-generation vital monitoring and AI predictive analytics powered by advanced wearable sensor technology.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FeatureCard 
          icon={Cpu}
          title="Multi-Sensor Fusion"
          desc="Combining Photoplethysmography (BVP), Electrodermal Activity (EDA), and precise skin thermometry to build a complete physiological profile in real-time."
          color="text-primary"
          bg="bg-primary/10"
        />
        <FeatureCard 
          icon={Brain}
          title="Predictive AI Engine"
          desc="Machine learning algorithms analyze vital correlations to predict fatigue, stress onset, and potential fever risks before symptoms manifest physically."
          color="text-secondary"
          bg="bg-secondary/10"
        />
        <FeatureCard 
          icon={Shield}
          title="Anomaly Detection"
          desc="Continuous baseline monitoring instantly flags deviations. Tachycardia, abnormal stress responses, and dangerous temperature spikes trigger immediate alerts."
          color="text-emerald-500"
          bg="bg-emerald-500/10"
        />
        <FeatureCard 
          icon={Watch}
          title="Interactive Visualization"
          desc="A complete 3D digital twin of the hardware allows you to visualize health data mapped directly onto the device interface contextually."
          color="text-orange-500"
          bg="bg-orange-500/10"
        />
      </div>

      <div className="glass-panel rounded-3xl p-8 mt-12">
        <h2 className="text-2xl font-bold mb-4">How it works</h2>
        <div className="space-y-4 text-muted-foreground leading-relaxed">
          <p>
            The manual input controls on the dashboard simulate raw data streams that would normally be transmitted via Bluetooth Low Energy (BLE) from the physical watch.
          </p>
          <p>
            When you adjust a slider, the system recalculates your <strong>Urgency Score</strong> and <strong>Severity Level</strong>. The urgency score is a weighted sum of deviations from your personal healthy baseline (BVP: 75bpm, EDA: 2.0µS, Temp: 36.6°C).
          </p>
          <p>
            Try pushing the EDA (Stress) slider up while keeping Activity at "Resting" to see the AI detect an "Abnormal Stress Response" anomaly.
          </p>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, desc, color, bg }: any) {
  return (
    <div className="glass-panel rounded-3xl p-6 hover:-translate-y-1 transition-transform duration-300">
      <div className={`w-12 h-12 rounded-2xl ${bg} flex items-center justify-center mb-4`}>
        <Icon className={`w-6 h-6 ${color}`} />
      </div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-muted-foreground text-sm leading-relaxed">{desc}</p>
    </div>
  );
}
