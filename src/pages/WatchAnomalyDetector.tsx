import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Heart, Activity, Wifi, WifiOff, Upload,
  FileText, CheckCircle, AlertTriangle, RotateCcw, ChevronRight, X
} from "lucide-react";

// ── Constants ─────────────────────────────────────────────────────────────────

const API_CSV    = "http://127.0.0.1:8000/predict/csv";
const API_HEALTH = "http://127.0.0.1:8000/health";
const TIMESTEPS  = 27;
const CHANNELS   = 7;
const CH_LABELS  = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z", "HR"];
const CH_COLORS  = ["#f43f5e","#06b6d4","#f97316","#a855f7","#8b5cf6","#3b82f6","#10b981"];

// ── Types ─────────────────────────────────────────────────────────────────────

interface PredictionResult {
  anomaly:        number;
  label:          string;
  score:          number;
  threshold:      number;
  confidence:     number;
  per_step_scores: number[];
  timesteps:      number;
  channels:       number;
  rows_received:  number;
  cols_received:  number;
  error?:         string;
}

interface ParsedCSV {
  headers: string[];
  rows:    number[][];
  raw:     string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function parseCSVPreview(text: string): ParsedCSV {
  const lines   = text.trim().split(/\r?\n/).filter(Boolean);
  const first   = lines[0].split(",");
  const hasHdr  = isNaN(parseFloat(first[0].trim()));
  const headers = hasHdr ? first.map(h => h.trim()) : CH_LABELS.slice(0, first.length);
  const dataLines = hasHdr ? lines.slice(1) : lines;
  const rows = dataLines.map(l => l.split(",").map(v => parseFloat(v.trim())));
  return { headers, rows, raw: text };
}

// ── ECG animation ─────────────────────────────────────────────────────────────

function ECGLine({ anomaly }: { anomaly: boolean }) {
  const ref    = useRef<HTMLCanvasElement>(null);
  const off    = useRef(0);
  const frame  = useRef(0);
  const color  = anomaly ? "#ef4444" : "#10b981";
  const path   = [0,0,0,0,0,-12,16,-2,0,0,0,3,0,0,0,0];

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height, seg = W / path.length;
    function draw() {
      ctx.clearRect(0,0,W,H);
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.shadowColor = color + "66"; ctx.shadowBlur = 5;
      ctx.beginPath();
      for (let i = 0; i < W + seg; i++) {
        const idx = Math.floor(((i - off.current % W) / W) * path.length + path.length) % path.length;
        i === 0 ? ctx.moveTo(i, H/2 + path[idx]) : ctx.lineTo(i, H/2 + path[idx]);
      }
      ctx.stroke();
      off.current += 1.5;
      frame.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(frame.current);
  }, [color]);

  return <canvas ref={ref} width={140} height={26} style={{ width:"100%", height:26 }} />;
}

// ── Watch face ────────────────────────────────────────────────────────────────

interface WatchProps {
  result:  PredictionResult | null;
  loading: boolean;
  color:   { body: string; strap: string };
}

function Watch({ result, loading, color }: WatchProps) {
  const isAnomaly  = result?.anomaly === 1;
  const ringColor  = loading ? "#f59e0b" : isAnomaly ? "#ef4444" : result ? "#10b981" : "#6366f1";
  const statusText = loading ? "SCANNING" : isAnomaly ? "ANOMALY" : result ? "NORMAL" : "IDLE";
  const [tick, setTick] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setTick(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ position:"relative", width:180, height:310, flexShrink:0 }}>
      {/* straps */}
      <div style={{ position:"absolute", width:80, height:95, top:0, left:"50%", marginLeft:-40, borderRadius:"10px 10px 0 0", background:color.strap }} />
      <div style={{ position:"absolute", width:80, height:95, bottom:0, left:"50%", marginLeft:-40, borderRadius:"0 0 10px 10px", background:color.strap }} />

      {/* glow ring */}
      <motion.div
        style={{ position:"absolute", width:146, height:168, top:"50%", left:"50%", marginLeft:-73, marginTop:-84,
          borderRadius:32, border:`1.5px solid ${ringColor}`, boxShadow:`0 0 20px ${ringColor}55`, zIndex:10, pointerEvents:"none" }}
        animate={{ rotate:360 }}
        transition={{ duration: loading ? 1.2 : 10, repeat:Infinity, ease:"linear" }}
      />

      {/* body */}
      <div style={{ position:"absolute", width:138, height:162, top:"50%", left:"50%", marginLeft:-69, marginTop:-81,
        borderRadius:28, background:color.body, zIndex:20,
        boxShadow:"0 8px 28px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.12)",
        display:"flex", alignItems:"center", justifyContent:"center" }}>

        {/* screen */}
        <div style={{ width:120, height:148, borderRadius:22, overflow:"hidden",
          background:"linear-gradient(160deg,#060610 0%,#0d0d1a 100%)",
          display:"flex", flexDirection:"column", padding:"5px 0 3px" }}>

          {/* time + status */}
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"0 7px", marginBottom:3 }}>
            <span style={{ fontSize:10, color:"#475569", fontFamily:"monospace" }}>
              {tick.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}
            </span>
            <motion.span
              animate={{ opacity: loading ? [1,0.3,1] : 1 }}
              transition={{ repeat: loading ? Infinity : 0, duration:0.8 }}
              style={{ fontSize:7, fontWeight:700, letterSpacing:1, padding:"2px 5px", borderRadius:6,
                color:ringColor, background:ringColor+"22" }}>
              {statusText}
            </motion.span>
          </div>

          {/* big score */}
          <div style={{ padding:"2px 7px", marginBottom:2 }}>
            <div style={{ fontSize:8, color:"#334155", marginBottom:1 }}>MSE SCORE</div>
            <div style={{ fontSize:20, fontWeight:700, fontFamily:"monospace", color: result ? ringColor : "#1e293b", lineHeight:1 }}>
              {result ? result.score.toFixed(4) : "-.----"}
            </div>
          </div>

          {/* stats grid */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:3, padding:"0 5px" }}>
            {[
              { l:"THRESH", v: result ? result.threshold.toFixed(3) : "–", c:"#64748b" },
              { l:"CONF",   v: result ? Math.round(result.confidence*100)+"%" : "–", c:ringColor },
              { l:"ROWS",   v: result ? String(result.rows_received) : "–", c:"#64748b" },
              { l:"COLS",   v: result ? String(result.cols_received) : "–", c:"#64748b" },
            ].map(item => (
              <div key={item.l} style={{ background:"#ffffff08", borderRadius:6, padding:"3px 4px" }}>
                <div style={{ fontSize:6, color:"#334155" }}>{item.l}</div>
                <div style={{ fontSize:10, fontWeight:600, fontFamily:"monospace", color:item.c }}>{item.v}</div>
              </div>
            ))}
          </div>

          {/* ECG */}
          <div style={{ padding:"3px 4px", marginTop:"auto" }}>
            <ECGLine anomaly={isAnomaly} />
          </div>
        </div>
      </div>

      {/* crown */}
      <div style={{ position:"absolute", width:6, height:24, right:10, top:"50%", marginTop:-40, zIndex:30,
        borderRadius:3, background:color.body, boxShadow:"2px 0 6px rgba(0,0,0,0.4)" }} />
    </div>
  );
}

// ── Sparkline chart ────────────────────────────────────────────────────────────

function SparkChart({ scores, threshold }: { scores: number[]; threshold: number }) {
  const max = Math.max(...scores, threshold * 1.5);
  const W = 100, H = 40;
  const pts = scores.map((s, i) => ({
    x: (i / (scores.length - 1)) * W,
    y: H - (s / max) * H,
    anomaly: s > threshold,
  }));
  const pathD = pts.map((p,i) => `${i===0?"M":"L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const thY   = H - (threshold / max) * H;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width:"100%", height:56 }}>
      {/* threshold line */}
      <line x1={0} y1={thY} x2={W} y2={thY} stroke="#f59e0b" strokeWidth={0.5} strokeDasharray="2,2" />
      {/* area fill */}
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${pathD} L${W},${H} L0,${H} Z`} fill="url(#sg)" />
      {/* line */}
      <path d={pathD} fill="none" stroke="#6366f1" strokeWidth={1.2} />
      {/* anomaly dots */}
      {pts.filter(p => p.anomaly).map((p,i) => (
        <circle key={i} cx={p.x} cy={p.y} r={2} fill="#ef4444" />
      ))}
    </svg>
  );
}

// ── CSV preview table ──────────────────────────────────────────────────────────

function CSVPreview({ parsed }: { parsed: ParsedCSV }) {
  const show = parsed.rows.slice(0, 6);
  return (
    <div style={{ overflowX:"auto", fontSize:10 }}>
      <table style={{ borderCollapse:"collapse", width:"100%", minWidth:300 }}>
        <thead>
          <tr>
            <th style={{ padding:"2px 5px", color:"#475569", textAlign:"left", borderBottom:"0.5px solid #1e293b" }}>#</th>
            {parsed.headers.map(h => (
              <th key={h} style={{ padding:"2px 5px", color:"#475569", textAlign:"right", borderBottom:"0.5px solid #1e293b", whiteSpace:"nowrap" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {show.map((row, i) => (
            <tr key={i}>
              <td style={{ padding:"2px 5px", color:"#334155", fontFamily:"monospace" }}>{i+1}</td>
              {row.map((v, j) => (
                <td key={j} style={{ padding:"2px 5px", color: CH_COLORS[j] ?? "#94a3b8", fontFamily:"monospace", textAlign:"right" }}>
                  {isNaN(v) ? "–" : v.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {parsed.rows.length > 6 && (
        <div style={{ color:"#334155", fontSize:9, marginTop:3, paddingLeft:5 }}>
          +{parsed.rows.length - 6} more rows (total {parsed.rows.length})
        </div>
      )}
    </div>
  );
}

// ── Channel mini-bars ──────────────────────────────────────────────────────────

function ChannelBars({ parsed }: { parsed: ParsedCSV }) {
  return (
    <div style={{ display:"grid", gridTemplateColumns:"repeat(7,1fr)", gap:4, marginTop:6 }}>
      {parsed.headers.slice(0, CHANNELS).map((h, ci) => {
        const vals = parsed.rows.map(r => r[ci]).filter(v => !isNaN(v));
        const mn   = Math.min(...vals), mx = Math.max(...vals);
        return (
          <div key={h} style={{ textAlign:"center" }}>
            <div style={{ fontSize:8, color: CH_COLORS[ci], marginBottom:2, fontWeight:600 }}>{h}</div>
            <div style={{ display:"flex", gap:1, height:24, alignItems:"flex-end", justifyContent:"center" }}>
              {vals.slice(-8).map((v, i) => (
                <div key={i} style={{ width:3, borderRadius:1, background: CH_COLORS[ci] + (i === vals.slice(-8).length-1 ? "ff" : "55"),
                  height: Math.max(2, ((v-mn)/(mx-mn||1))*22) }} />
              ))}
            </div>
            <div style={{ fontSize:7, color:"#475569", fontFamily:"monospace" }}>
              {isNaN(mn) ? "–" : mn.toFixed(1)}–{isNaN(mx) ? "–" : mx.toFixed(1)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function WatchAnomalyDetector() {
  const [result,    setResult]    = useState<PredictionResult | null>(null);
  const [loading,   setLoading]   = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [parsed,    setParsed]    = useState<ParsedCSV | null>(null);
  const [fileName,  setFileName]  = useState<string>("");
  const [dragOver,  setDragOver]  = useState(false);
  const [error,     setError]     = useState<string>("");
  const [watchColor]              = useState({ body:"#0f172a", strap:"#6366f1" });
  const fileRef = useRef<HTMLInputElement>(null);

  // health check
  useEffect(() => {
    const check = async () => {
      try { const r = await fetch(API_HEALTH, { signal: AbortSignal.timeout(2000) }); setBackendUp(r.ok); }
      catch { setBackendUp(false); }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.name.endsWith(".csv")) { setError("Please upload a .csv file"); return; }
    setError("");
    setFileName(file.name);
    setResult(null);
    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target?.result as string;
      try { setParsed(parseCSVPreview(text)); }
      catch { setError("Could not parse CSV"); }
    };
    reader.readAsText(file);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!parsed || !fileRef.current?.files?.[0]) return;
    setLoading(true); setError("");
    try {
      const form = new FormData();
      form.append("file", fileRef.current.files[0]);
      const res  = await fetch(API_CSV, { method:"POST", body:form, signal: AbortSignal.timeout(15000) });
      const data = await res.json() as PredictionResult;
      if (!res.ok) throw new Error((data as any).detail ?? "Server error");
      setResult(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg.includes("timeout") ? "Request timed out" : msg.includes("fetch") ? "Backend unreachable — is backend_server.py running?" : msg);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setParsed(null); setResult(null); setFileName(""); setError("");
    if (fileRef.current) fileRef.current.value = "";
  };

  const isAnomaly  = result?.anomaly === 1;
  const statusBg   = backendUp === true ? "#10b98115" : backendUp === false ? "#ef444415" : "#6366f115";
  const statusColor= backendUp === true ? "#10b981"   : backendUp === false ? "#ef4444"   : "#6366f1";

  return (
    <div style={{ fontFamily:"'DM Mono', 'Fira Mono', monospace", maxWidth:860, margin:"0 auto", padding:"1.5rem 1rem",
      background:"var(--color-background-primary)", minHeight:"100vh", color:"var(--color-text-primary)" }}>

      {/* ── header ── */}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:20 }}>
        <div>
          <div style={{ fontSize:11, letterSpacing:3, color:"#6366f1", marginBottom:4, textTransform:"uppercase" }}>PRATA v5 · Autoencoder</div>
          <h1 style={{ margin:0, fontSize:22, fontWeight:700, letterSpacing:-0.5 }}>Wristband Anomaly Detector</h1>
          <p style={{ margin:"4px 0 0", fontSize:12, color:"var(--color-text-secondary)" }}>
            Upload a CSV of health signal time-series → model detects pattern deviations
          </p>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:6, fontSize:11, padding:"5px 12px", borderRadius:99, background:statusBg, color:statusColor }}>
          {backendUp === true ? <Wifi size={11}/> : backendUp === false ? <WifiOff size={11}/> : <Activity size={11}/>}
          {backendUp === true ? "Backend online" : backendUp === false ? "Backend offline" : "Checking..."}
        </div>
      </div>

      {/* ── main grid ── */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 220px", gap:16 }}>

        {/* LEFT column */}
        <div style={{ display:"flex", flexDirection:"column", gap:12 }}>

          {/* drop zone */}
          <div
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileRef.current?.click()}
            style={{
              border: dragOver ? "1.5px dashed #6366f1" : parsed ? "1.5px solid #6366f140" : "1.5px dashed #1e293b",
              borderRadius:12, padding:"20px 16px", cursor:"pointer", textAlign:"center",
              background: dragOver ? "#6366f108" : parsed ? "#6366f105" : "transparent",
              transition:"all 0.15s"
            }}>
            <input ref={fileRef} type="file" accept=".csv" style={{ display:"none" }}
              onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }} />

            {parsed ? (
              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                  <FileText size={16} style={{ color:"#6366f1" }} />
                  <div style={{ textAlign:"left" }}>
                    <div style={{ fontSize:12, fontWeight:600 }}>{fileName}</div>
                    <div style={{ fontSize:10, color:"var(--color-text-secondary)" }}>
                      {parsed.rows.length} rows × {parsed.headers.length} channels
                    </div>
                  </div>
                </div>
                <button onClick={e => { e.stopPropagation(); reset(); }}
                  style={{ background:"none", border:"none", cursor:"pointer", color:"#475569", padding:4 }}>
                  <X size={14} />
                </button>
              </div>
            ) : (
              <>
                <Upload size={20} style={{ color:"#334155", marginBottom:6 }} />
                <div style={{ fontSize:13, fontWeight:500 }}>Drop your CSV here or click to browse</div>
                <div style={{ fontSize:11, color:"var(--color-text-secondary)", marginTop:4 }}>
                  27 rows × 7 columns · BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z, HR
                </div>
              </>
            )}
          </div>

          {/* CSV preview */}
          <AnimatePresence>
            {parsed && (
              <motion.div initial={{ opacity:0, y:6 }} animate={{ opacity:1, y:0 }}
                style={{ background:"var(--color-background-secondary)", border:"0.5px solid var(--color-border-tertiary)", borderRadius:10, padding:"10px 12px" }}>
                <div style={{ fontSize:10, color:"var(--color-text-secondary)", marginBottom:6, textTransform:"uppercase", letterSpacing:1 }}>Preview</div>
                <CSVPreview parsed={parsed} />
                <ChannelBars parsed={parsed} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* error */}
          {error && (
            <div style={{ background:"#ef444410", border:"0.5px solid #ef444430", borderRadius:8, padding:"8px 12px", fontSize:12, color:"#ef4444" }}>
              ❌ {error}
            </div>
          )}

          {/* analyze button */}
          <button onClick={analyze} disabled={!parsed || loading}
            style={{ padding:"10px 0", borderRadius:10, border:"none", fontSize:14, fontWeight:600,
              cursor: (!parsed || loading) ? "not-allowed" : "pointer",
              background: !parsed ? "#1e293b" : loading ? "#6366f188" : "#6366f1",
              color: !parsed ? "#475569" : "#fff",
              display:"flex", alignItems:"center", justifyContent:"center", gap:8, transition:"all 0.15s" }}>
            {loading ? (
              <>
                <motion.div animate={{ rotate:360 }} transition={{ repeat:Infinity, duration:1, ease:"linear" }}>
                  <Activity size={15} />
                </motion.div>
                Running inference...
              </>
            ) : (
              <><ChevronRight size={15} /> Analyze Signal Window</>
            )}
          </button>

          {/* result card */}
          <AnimatePresence>
            {result && (
              <motion.div key="result" initial={{ opacity:0, y:8 }} animate={{ opacity:1, y:0 }}
                style={{ background: isAnomaly ? "#ef444408" : "#10b98108",
                  border: `0.5px solid ${isAnomaly ? "#ef444430" : "#10b98130"}`,
                  borderRadius:10, padding:"12px 14px" }}>

                <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:10 }}>
                  {isAnomaly
                    ? <AlertTriangle size={18} style={{ color:"#ef4444" }} />
                    : <CheckCircle  size={18} style={{ color:"#10b981" }} />}
                  <span style={{ fontSize:16, fontWeight:700, color: isAnomaly ? "#ef4444" : "#10b981" }}>
                    {isAnomaly ? "Anomaly Detected" : "Normal Pattern"}
                  </span>
                </div>

                <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginBottom:10 }}>
                  {[
                    { l:"MSE Score",  v: result.score.toFixed(6)     },
                    { l:"Threshold",  v: result.threshold.toFixed(4)  },
                    { l:"Confidence", v: Math.round(result.confidence*100)+"%" },
                  ].map(m => (
                    <div key={m.l} style={{ background:"var(--color-background-secondary)", borderRadius:8, padding:"7px 9px" }}>
                      <div style={{ fontSize:9, color:"var(--color-text-secondary)", marginBottom:2 }}>{m.l}</div>
                      <div style={{ fontSize:13, fontWeight:600, fontFamily:"monospace" }}>{m.v}</div>
                    </div>
                  ))}
                </div>

                {/* confidence bar */}
                <div style={{ marginBottom:10 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", fontSize:9, color:"var(--color-text-secondary)", marginBottom:3 }}>
                    <span>Confidence</span><span>{Math.round(result.confidence*100)}%</span>
                  </div>
                  <div style={{ height:5, borderRadius:99, background:"var(--color-background-secondary)", overflow:"hidden" }}>
                    <motion.div initial={{ width:0 }} animate={{ width: Math.round(result.confidence*100)+"%" }}
                      style={{ height:"100%", borderRadius:99, background: isAnomaly ? "#ef4444" : "#10b981" }} />
                  </div>
                </div>

                {/* per-step sparkline */}
                {result.per_step_scores?.length > 0 && (
                  <div>
                    <div style={{ fontSize:9, color:"var(--color-text-secondary)", marginBottom:3 }}>
                      Per-timestep MSE · <span style={{ color:"#f59e0b" }}>— threshold</span>
                      {" · "}<span style={{ color:"#ef4444" }}>● anomaly steps</span>
                    </div>
                    <SparkChart scores={result.per_step_scores} threshold={result.threshold} />
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* RIGHT — watch */}
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:12 }}>
          <Watch result={result} loading={loading} color={watchColor} />

          {/* model info */}
          <div style={{ background:"var(--color-background-secondary)", border:"0.5px solid var(--color-border-tertiary)",
            borderRadius:10, padding:"10px 12px", width:"100%", boxSizing:"border-box" }}>
            <div style={{ fontSize:9, color:"var(--color-text-secondary)", marginBottom:6, textTransform:"uppercase", letterSpacing:1 }}>Model</div>
            {[
              ["Architecture", "Autoencoder"],
              ["Input",        "27×7 = 189"],
              ["Layers",       "Conv+LSTM+Transformer"],
              ["Decision",     "Recon. Error (MSE)"],
              ["Rules",        "None — pure ML"],
            ].map(([k,v]) => (
              <div key={k} style={{ display:"flex", justifyContent:"space-between", fontSize:10, marginBottom:4 }}>
                <span style={{ color:"var(--color-text-secondary)" }}>{k}</span>
                <span style={{ fontWeight:500, color: k==="Rules" ? "#10b981" : "var(--color-text-primary)" }}>{v}</span>
              </div>
            ))}
          </div>

          {/* CSV format guide */}
          <div style={{ background:"var(--color-background-secondary)", border:"0.5px solid var(--color-border-tertiary)",
            borderRadius:10, padding:"10px 12px", width:"100%", boxSizing:"border-box" }}>
            <div style={{ fontSize:9, color:"var(--color-text-secondary)", marginBottom:6, textTransform:"uppercase", letterSpacing:1 }}>CSV Format</div>
            <div style={{ fontSize:9, fontFamily:"monospace", color:"#475569", lineHeight:1.8 }}>
              {CH_LABELS.map((l,i) => (
                <div key={l} style={{ color: CH_COLORS[i] }}>{l}</div>
              ))}
            </div>
            <div style={{ fontSize:9, color:"var(--color-text-secondary)", marginTop:6 }}>
              27 rows minimum · header optional
            </div>
          </div>
        </div>
      </div>

      {/* offline banner */}
      {backendUp === false && (
        <div style={{ marginTop:14, padding:"9px 13px", borderRadius:8, fontSize:11, fontFamily:"monospace",
          background:"#ef444410", border:"0.5px solid #ef444430", color:"#ef4444" }}>
          Backend offline → <strong>python backend_server.py</strong> (keep Terminal 1 running)
        </div>
      )}
    </div>
  );
}
