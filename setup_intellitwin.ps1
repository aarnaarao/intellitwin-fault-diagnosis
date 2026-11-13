# IntelliTwin one-shot setup for Windows PowerShell
# Run from the root of intellitwin-fault-diagnosis:
# powershell -ExecutionPolicy Bypass -File .\setup_intellitwin.ps1

$ErrorActionPreference = "Stop"
$Root = (Get-Location).Path

if (!(Test-Path (Join-Path $Root "results\intellitwin_model.joblib"))) {
    throw "Could not find results\intellitwin_model.joblib. Run this script from the project root after training the model."
}
if (!(Test-Path (Join-Path $Root "data\pump_predictive_maintenance_dataset1.csv"))) {
    throw "Could not find data\pump_predictive_maintenance_dataset1.csv. Run this script from the project root."
}
if (!(Test-Path (Join-Path $Root "src\faultdiag\features.py"))) {
    throw "Could not find src\faultdiag\features.py. This does not look like the expected IntelliTwin project."
}

Write-Host "`n[1/5] Creating FastAPI backend..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "backend" | Out-Null
@'
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faultdiag.data import load_data
from faultdiag.features import build_features

MODEL_PATH = ROOT / "results" / "intellitwin_model.joblib"
DATA_PATH = ROOT / "data" / "pump_predictive_maintenance_dataset1.csv"
RESULTS_PATH = ROOT / "results" / "results.csv"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model bundle not found: {MODEL_PATH}")
if not DATA_PATH.exists():
    raise RuntimeError(f"Dataset not found: {DATA_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
classes = bundle.get("classes", ["Normal", "Overload", "Pressure_drop", "Shutdown"])
best_model = bundle.get("best_model", "xgboost")
best_strategy = bundle.get("best_strategy", "smote")

# Use the exact training feature builder and saved column order.
raw_df = load_data(str(DATA_PATH))
processed_df, feature_frame = build_features(raw_df, columns=feature_columns)
scaled_features = scaler.transform(feature_frame)
latest_indices = processed_df.groupby("asset_id", sort=False).tail(1).index
latest_df = processed_df.loc[latest_indices].copy()
latest_df = latest_df.sort_values("asset_id").reset_index(drop=True)
latest_index_by_asset = {str(processed_df.loc[int(i), "asset_id"]): int(i) for i in latest_indices}

app = FastAPI(title="IntelliTwin API", version="1.0.0",
              description="Pump fault diagnosis using the saved IntelliTwin XGBoost model.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def row_payload(row):
    return {
        "asset_id": str(row["asset_id"]),
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "pump_type": str(row.get("pump_type", "Unknown")),
        "manufacturer": str(row.get("manufacturer", "Unknown")),
        "flow_rate_bpd": safe_float(row.get("flow_rate_bpd")),
        "pressure_psi": safe_float(row.get("pressure_psi")),
        "temperature_c": safe_float(row.get("temperature_c")),
        "vibration_mm_s": safe_float(row.get("vibration_mm_s")),
        "rotation_rpm": safe_float(row.get("rotation_rpm")),
        "power_kw": safe_float(row.get("power_kw")),
        "current_a": safe_float(row.get("current_a")),
        "voltage_v": safe_float(row.get("voltage_v")),
        "efficiency_pct": safe_float(row.get("efficiency_pct")),
        "operating_hours": safe_float(row.get("operating_hours")),
    }

def prediction_for_index(index):
    x = scaled_features[index:index + 1]
    predicted = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    probability_map = {str(label): round(float(prob), 6)
                       for label, prob in zip(model.classes_, probabilities)}
    confidence = float(np.max(probabilities))
    fault = str(predicted)
    if fault == "Normal":
        risk = "Low"
    elif confidence >= 0.80:
        risk = "High"
    else:
        risk = "Elevated"
    return {
        "predicted_fault": fault,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "risk_category": risk,
        "class_probabilities": probability_map,
        "model": best_model,
        "imbalance_strategy": best_strategy,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(feature_columns)}

@app.get("/api/overview")
def overview():
    counts = latest_df.shape[0]
    machine_payloads = []
    for _, row in latest_df.iterrows():
        idx = int(row.name) if row.name is not None else None
        machine_payloads.append(row_payload(row))
    # Predictions are calculated against the exact latest-row feature index.
    fault_counts = {c: 0 for c in classes}
    for idx in latest_indices:
        pred = prediction_for_index(int(idx))["predicted_fault"]
        fault_counts[pred] = fault_counts.get(pred, 0) + 1
    return {
        "machine_count": int(counts),
        "fault_counts": fault_counts,
        "latest_timestamp": pd.Timestamp(latest_df["timestamp"].max()).isoformat(),
        "model_name": best_model,
        "strategy": best_strategy,
        "feature_count": len(feature_columns),
        "data_mode": "Historical dataset",
    }

@app.get("/api/machines")
def machines():
    items = []
    for idx in latest_indices:
        row = processed_df.loc[int(idx)]
        item = row_payload(row)
        item.update(prediction_for_index(int(idx)))
        items.append(item)
    return {"machines": items}

@app.get("/api/machines/{asset_id}")
def machine_detail(asset_id: str):
    matches = latest_df[latest_df["asset_id"].astype(str) == str(asset_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Machine not found")
    idx = latest_index_by_asset[str(asset_id)]
    row = processed_df.loc[idx]
    result = row_payload(row)
    result.update(prediction_for_index(idx))
    return result

@app.get("/api/diagnose/{asset_id}")
def diagnose(asset_id: str):
    matches = latest_df[latest_df["asset_id"].astype(str) == str(asset_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Machine not found")
    idx = latest_index_by_asset[str(asset_id)]
    row = processed_df.loc[idx]
    result = row_payload(row)
    result.update(prediction_for_index(idx))
    result["diagnosis_basis"] = "Latest available historical reading for this pump"
    return result

@app.get("/api/model-results")
def model_results():
    if not RESULTS_PATH.exists():
        return {"rows": []}
    frame = pd.read_csv(RESULTS_PATH)
    rows = frame.replace({np.nan: None}).to_dict(orient="records")
    return {"rows": rows}
'@ | Set-Content -Encoding UTF8 "backend\main.py"

Write-Host "[2/5] Creating React application..." -ForegroundColor Cyan
if (Test-Path "frontend") {
    $backup = "frontend_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Copy-Item "frontend" $backup -Recurse
    Write-Host "Existing frontend backed up to $backup" -ForegroundColor Yellow
}
New-Item -ItemType Directory -Force -Path "frontend\src" | Out-Null
@'
{
  "name": "intellitwin-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": { "dev": "vite --host 127.0.0.1" },
  "dependencies": {
    "framer-motion": "^11.11.17",
    "lucide-react": "^0.468.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.15.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^5.4.11"
  }
}
'@ | Set-Content -Encoding UTF8 "frontend\package.json"
@'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } }
});
'@ | Set-Content -Encoding UTF8 "frontend\vite.config.js"
@'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#080d16" />
    <title>IntelliTwin | Predictive Intelligence</title>
  </head>
  <body><div id="root"></div><script type="module" src="/src/main.jsx"></script></body>
</html>
'@ | Set-Content -Encoding UTF8 "frontend\index.html"
@'
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./style.css";
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>
);
'@ | Set-Content -Encoding UTF8 "frontend\src\main.jsx"

@'
import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity, AlertTriangle, ArrowUpRight, BarChart3, Bell, CheckCircle2,
  ChevronDown, CircleGauge, Cpu, Factory, Gauge, LayoutDashboard, Menu,
  Search, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Waves,
  Zap, Thermometer, RotateCw, RefreshCw, X
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Cell, Bar, BarChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from "recharts";

const API = "";
const fmt = (v, digits=1) => v == null ? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:digits});
const labelize = (s="") => s.replaceAll("_"," ").replace(/\b\w/g, c=>c.toUpperCase());
const palette = {Normal:"#36c99a", Overload:"#f4b64f", Pressure_drop:"#65a8ff", Shutdown:"#ff6c78"};
const riskColor = r => r==="Low" ? "green" : r==="High" ? "red" : "amber";

function App(){
  const [page,setPage] = useState("Overview");
  const [machines,setMachines] = useState([]);
  const [results,setResults] = useState([]);
  const [overview,setOverview] = useState(null);
  const [selected,setSelected] = useState("");
  const [diagnosis,setDiagnosis] = useState(null);
  const [loading,setLoading] = useState(true);
  const [diagnosing,setDiagnosing] = useState(false);
  const [error,setError] = useState("");
  const [query,setQuery] = useState("");
  const [mobileNav,setMobileNav] = useState(false);

  const load = async () => {
    setLoading(true); setError("");
    try {
      const [m,o,r] = await Promise.all([
        fetch(`${API}/api/machines`), fetch(`${API}/api/overview`), fetch(`${API}/api/model-results`)
      ]);
      if(!m.ok || !o.ok || !r.ok) throw new Error("The IntelliTwin API did not return a successful response.");
      const [md,od,rd] = await Promise.all([m.json(),o.json(),r.json()]);
      setMachines(md.machines || []); setOverview(od); setResults(rd.rows || []);
      setSelected(prev => prev || md.machines?.[0]?.asset_id || "");
    } catch(e){ setError(e.message + " Check that the backend window is running."); }
    finally { setLoading(false); }
  };
  useEffect(()=>{load();},[]);
  const active = machines.find(m=>m.asset_id===selected) || machines[0];
  const filtered = useMemo(()=>machines.filter(m=>m.asset_id.toLowerCase().includes(query.toLowerCase())),[machines,query]);
  const bestRows = useMemo(()=>results.slice().sort((a,b)=>(b.val_macro_f1||0)-(a.val_macro_f1||0)),[results]);
  const diagnose = async () => {
    if(!selected) return;
    setDiagnosing(true); setDiagnosis(null);
    try{
      const res=await fetch(`/api/diagnose/${encodeURIComponent(selected)}`);
      const data=await res.json();
      if(!res.ok) throw new Error(data.detail || "Diagnosis failed");
      setDiagnosis(data); setPage("Diagnosis");
    }catch(e){setError(e.message);}
    finally{setDiagnosing(false);}
  };
  const nav = [
    {name:"Overview",icon:LayoutDashboard},
    {name:"Machine Fleet",icon:Factory},
    {name:"Fault Diagnosis",icon:Activity},
    {name:"Model Intelligence",icon:BarChart3}
  ];
  const faultTotal = overview ? Object.entries(overview.fault_counts||{}).filter(([k])=>k!=="Normal").reduce((a,[,v])=>a+v,0):0;
  const chartData = Object.entries(overview?.fault_counts||{}).map(([name,value])=>({name:labelize(name),count:value,fill:palette[name]}));
  const metricCards = [
    {label:"Monitored assets",value:overview?.machine_count ?? "—",sub:"Pumps in dataset",icon:Factory,tone:"blue"},
    {label:"Predicted fault states",value:faultTotal,sub:"Latest reading per pump",icon:AlertTriangle,tone:"amber"},
    {label:"Healthy predictions",value:overview?.fault_counts?.Normal ?? "—",sub:"Classified as normal",icon:ShieldCheck,tone:"green"},
    {label:"Input features",value:overview?.feature_count ?? "—",sub:"Training-aligned features",icon:Cpu,tone:"violet"}
  ];
  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?"show":""}`}>
      <div className="brand"><div className="brand-mark"><Activity size={21}/></div><div><b>intelli<span>twin</span></b><small>INDUSTRIAL INTELLIGENCE</small></div><button className="close-nav" onClick={()=>setMobileNav(false)}><X size={18}/></button></div>
      <div className="workspace"><span className="workspace-dot"/><div><b>Operations workspace</b><small>Predictive maintenance</small></div><ChevronDown size={15}/></div>
      <div className="nav-label">WORKSPACE</div>
      <nav>{nav.map(n=><button key={n.name} className={`nav-link ${page===n.name?"active":""}`} onClick={()=>{setPage(n.name);setMobileNav(false);}}><n.icon size={18}/><span>{n.name}</span>{n.name==="Fault Diagnosis"&&<i/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="system-card"><div className="system-head"><span className="live-dot"/> SYSTEM STATUS</div><b>Model operational</b><small>XGBoost · SMOTE balanced</small><div className="system-line"><span/></div><small>Prediction service connected</small></div><button className="nav-link"><Settings2 size={18}/>Settings</button><div className="user-row"><div className="avatar">IT</div><div><b>IntelliTwin Admin</b><small>Project demonstrator</small></div><ChevronDown size={15}/></div></div>
    </aside>
    {mobileNav&&<div className="scrim" onClick={()=>setMobileNav(false)}/>}
    <main className="main">
      <header className="topbar"><div className="top-left"><button className="mobile-menu" onClick={()=>setMobileNav(true)}><Menu/></button><div className="crumb">Workspace <span>/</span> <b>{page}</b></div></div><div className="top-actions"><div className="data-status"><span className="live-dot"/> MODEL CONNECTED</div><button className="icon-btn" onClick={load} title="Refresh data"><RefreshCw size={17}/></button><button className="icon-btn"><Bell size={17}/><i/></button><div className="top-avatar">IT</div></div></header>
      <div className="content">
        {error&&<div className="error-banner"><AlertTriangle size={17}/>{error}<button onClick={()=>setError("")}><X size={16}/></button></div>}
        {loading?<div className="loading"><div className="loader"/><b>Connecting to IntelliTwin services...</b><span>Loading pump telemetry and model performance</span></div>:<>
        {page==="Overview"&&<motion.div initial={{opacity:0,y:8}} animate={{opacity:1,y:0}}>
          <div className="page-heading"><div><div className="eyebrow"><Sparkles size={13}/> INTELLIGENT ASSET MONITORING</div><h1>Command center</h1><p>Operational visibility across your pump fleet, powered by machine learning.</p></div><button className="secondary-btn" onClick={load}><RefreshCw size={15}/> Refresh data</button></div>
          <div className="hero"><div className="hero-copy"><div className="hero-kicker"><span/> INTELLITWIN PREDICTIVE ENGINE</div><h2>See the signal.<br/><em>Predict the failure.</em></h2><p>AI-assisted fault classification built on historical pump telemetry and a trained XGBoost model.</p><button className="hero-btn" onClick={()=>setPage("Fault Diagnosis")}>Open fault diagnosis <ArrowUpRight size={16}/></button></div><div className="hero-visual"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="core"><Activity size={48}/></div><div className="float-tag tag-a"><span className="green-dot"/> MODEL ONLINE</div><div className="float-tag tag-b"><Waves size={14}/> SENSOR DATA</div><div className="visual-bars">{[32,48,40,70,55,86,62,78,46,91,66,76,54,88].map((v,i)=><i key={i} style={{height:v+"%"}}/>)}</div></div><div className="hero-foot"><span><span className="green-dot"/> MODEL CONNECTED</span><span>39 engineered features</span><span>4 fault classes</span></div></div>
          <div className="metric-grid">{metricCards.map((m,i)=><motion.div className="metric-card" key={m.label} initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:i*.06}}><div className="metric-top"><span>{m.label}</span><div className={`metric-icon ${m.tone}`}><m.icon size={18}/></div></div><div className="metric-value">{m.value}</div><div className="metric-sub"><span className="tiny-dot"/> {m.sub}</div></motion.div>)}</div>
          <div className="section-grid"><section className="panel"><div className="panel-heading"><div><h3>Fleet diagnosis distribution</h3><p>Model predictions from each pump's latest historical reading</p></div><span className="pill">50 ASSETS</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{top:10,right:8,left:-20,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke="#202b3b" vertical={false}/><XAxis dataKey="name" tick={{fill:"#8290a5",fontSize:11}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#8290a5",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111a28",border:"1px solid #263449",borderRadius:10,color:"#eaf0f8"}}/><Bar dataKey="count" radius={[5,5,0,0]}>{chartData.map((d,i)=><Cell key={i} fill={d.fill}/>)}</Bar></BarChart></ResponsiveContainer></div></section>
          <section className="panel"><div className="panel-heading"><div><h3>Quick diagnosis</h3><p>Run the trained model on a pump's latest reading</p></div><div className="panel-icon"><CircleGauge size={18}/></div></div><label className="field-label">SELECT ASSET</label><div className="select-wrap"><select value={selected} onChange={e=>setSelected(e.target.value)}>{machines.map(m=><option key={m.asset_id} value={m.asset_id}>{m.asset_id} · {m.pump_type}</option>)}</select><ChevronDown size={16}/></div>{active&&<div className="quick-sensors"><div><span>Pressure</span><b>{fmt(active.pressure_psi)} <small>psi</small></b></div><div><span>Flow rate</span><b>{fmt(active.flow_rate_bpd)} <small>bpd</small></b></div><div><span>Vibration</span><b>{fmt(active.vibration_mm_s)} <small>mm/s</small></b></div></div>}<button className="primary-btn full" disabled={!selected||diagnosing} onClick={diagnose}>{diagnosing?"Analyzing telemetry...": "Run AI diagnosis"} <ArrowUpRight size={16}/></button><div className="helper-note"><ShieldCheck size={14}/> Uses saved XGBoost model and original feature pipeline</div></section></div>
          <div className="section-heading"><div><h3>Priority assets</h3><p>Latest model classification across selected machines</p></div><button className="text-btn" onClick={()=>setPage("Machine Fleet")}>View all assets <ArrowUpRight size={15}/></button></div><div className="table-panel"><table><thead><tr><th>ASSET</th><th>PUMP TYPE</th><th>PRESSURE</th><th>FLOW RATE</th><th>MODEL CLASS</th><th>CONFIDENCE</th></tr></thead><tbody>{machines.slice().sort((a,b)=>(b.risk_category==="High")-(a.risk_category==="High")).slice(0,5).map(m=><tr key={m.asset_id} onClick={()=>{setSelected(m.asset_id);setPage("Machine Fleet");}}><td><b className="asset-id">{m.asset_id}</b></td><td>{m.pump_type}</td><td>{fmt(m.pressure_psi)} psi</td><td>{fmt(m.flow_rate_bpd)} bpd</td><td><span className="fault-label" style={{"--fault":palette[m.predicted_fault]}}>{labelize(m.predicted_fault)}</span></td><td><b>{m.confidence_percent}%</b></td></tr>)}</tbody></table></div>
        </motion.div>}
        {page==="Machine Fleet"&&<motion.div initial={{opacity:0}} animate={{opacity:1}}><div className="page-heading"><div><div className="eyebrow">ASSET INTELLIGENCE</div><h1>Machine fleet</h1><p>Explore all pump assets and their latest model-derived condition.</p></div><div className="searchbox"><Search size={16}/><input placeholder="Search asset ID..." value={query} onChange={e=>setQuery(e.target.value)}/></div></div><div className="fleet-grid">{filtered.map(m=><button className={`fleet-card ${selected===m.asset_id?"chosen":""}`} key={m.asset_id} onClick={()=>setSelected(m.asset_id)}><div className="fleet-card-top"><div className="pump-icon"><Factory size={20}/></div><span className={`risk-chip ${riskColor(m.risk_category)}`}>{m.risk_category} risk</span></div><b>{m.asset_id}</b><small>{m.pump_type} · {m.manufacturer}</small><div className="fleet-readings"><span>PRESSURE<strong>{fmt(m.pressure_psi)} psi</strong></span><span>VIBRATION<strong>{fmt(m.vibration_mm_s)} mm/s</strong></span></div><div className="fleet-bottom"><span className="fault-label" style={{"--fault":palette[m.predicted_fault]}}>{labelize(m.predicted_fault)}</span><span>{m.confidence_percent}% confidence</span></div></button>)}</div>{active&&<div className="selected-action"><div><b>{active.asset_id} selected</b><span>Run diagnosis on the latest historical reading.</span></div><button className="primary-btn" onClick={diagnose}>Diagnose asset <ArrowUpRight size={16}/></button></div>}</motion.div>}
        {page==="Fault Diagnosis"&&<motion.div initial={{opacity:0}} animate={{opacity:1}}><div className="page-heading"><div><div className="eyebrow">AI-POWERED CLASSIFICATION</div><h1>Fault diagnosis</h1><p>Classify the latest available pump reading with the saved XGBoost model.</p></div><div className="model-tag"><span className="green-dot"/> XGBOOST MODEL</div></div><div className="diagnosis-layout"><section className="panel diagnosis-form"><div className="panel-heading"><div><h3>Diagnosis request</h3><p>Choose a pump from the historical dataset</p></div><SlidersHorizontal size={19}/></div><label className="field-label">PUMP ASSET</label><div className="select-wrap"><select value={selected} onChange={e=>{setSelected(e.target.value);setDiagnosis(null);}}>{machines.map(m=><option key={m.asset_id} value={m.asset_id}>{m.asset_id} · {m.pump_type}</option>)}</select><ChevronDown size={16}/></div>{active&&<div className="sensor-grid">{[{n:"Flow rate",v:fmt(active.flow_rate_bpd),u:"bpd",i:Waves},{n:"Pressure",v:fmt(active.pressure_psi),u:"psi",i:Gauge},{n:"Temperature",v:fmt(active.temperature_c),u:"°C",i:Thermometer},{n:"Vibration",v:fmt(active.vibration_mm_s),u:"mm/s",i:Activity},{n:"Rotation",v:fmt(active.rotation_rpm),u:"rpm",i:RotateCw},{n:"Power",v:fmt(active.power_kw),u:"kW",i:Zap}].map(s=><div className="sensor-box" key={s.n}><s.i size={16}/><span>{s.n}</span><b>{s.v} <small>{s.u}</small></b></div>)}</div>}<button className="primary-btn full diagnose-button" onClick={diagnose} disabled={diagnosing}>{diagnosing?<><RefreshCw className="spin" size={16}/> Analyzing...</>:<><Sparkles size={16}/> Analyze machine condition</>}</button><p className="helper-note"><ShieldCheck size={14}/> No synthetic sensor readings. Original engineered features are reused.</p></section><section className="panel result-panel"><div className="panel-heading"><div><h3>Diagnosis result</h3><p>Prediction and class probability distribution</p></div><div className="panel-icon"><Activity size={18}/></div></div>{(diagnosis||active)?<><div className="result-placeholder">{diagnosis?<><div className="result-overline">PREDICTED FAULT CLASS</div><div className="result-name" style={{color:palette[diagnosis.predicted_fault]}}>{labelize(diagnosis.predicted_fault)}</div><div className={`risk-chip ${riskColor(diagnosis.risk_category)}`}>{diagnosis.risk_category} risk category</div><div className="confidence-row"><span>Model confidence</span><b>{diagnosis.confidence_percent}%</b></div><div className="confidence-track"><i style={{width:diagnosis.confidence_percent+"%",background:palette[diagnosis.predicted_fault]}}/></div><div className="prob-list">{Object.entries(diagnosis.class_probabilities).map(([k,v])=><div className="prob-row" key={k}><span>{labelize(k)}</span><div><i style={{width:(v*100)+"%",background:palette[k]}}/></div><b>{(v*100).toFixed(1)}%</b></div>)}</div><div className="result-disclaimer">Confidence is the classifier's predicted class probability, not a calibrated probability of future failure. {diagnosis.diagnosis_basis}.</div></>:<div className="empty-result"><div className="empty-icon"><CircleGauge size={30}/></div><b>Ready to analyze</b><span>Select a pump and run diagnosis to see its predicted fault, model confidence and class probabilities.</span><button className="secondary-btn" onClick={diagnose}>Run diagnosis <ArrowUpRight size={15}/></button></div>}</div></>:null}</section></div></motion.div>}
        {page==="Model Intelligence"&&<motion.div initial={{opacity:0}} animate={{opacity:1}}><div className="page-heading"><div><div className="eyebrow">MODEL PERFORMANCE & EVALUATION</div><h1>Model intelligence</h1><p>Validation and test metrics from the completed experiment runs.</p></div><div className="model-tag"><span className="green-dot"/> BEST MODEL · XGBOOST</div></div><div className="metric-grid compact"><div className="metric-card"><div className="metric-top"><span>Test macro F1</span><div className="metric-icon blue"><BarChart3 size={18}/></div></div><div className="metric-value">0.875</div><div className="metric-sub">SMOTE + XGBoost</div></div><div className="metric-card"><div className="metric-top"><span>Test accuracy</span><div className="metric-icon green"><CheckCircle2 size={18}/></div></div><div className="metric-value">98.7%</div><div className="metric-sub">Held-out pump assets</div></div><div className="metric-card"><div className="metric-top"><span>Model families</span><div className="metric-icon violet"><Cpu size={18}/></div></div><div className="metric-value">3</div><div className="metric-sub">Random Forest · SVM · XGBoost</div></div><div className="metric-card"><div className="metric-top"><span>Strategies</span><div className="metric-icon amber"><SlidersHorizontal size={18}/></div></div><div className="metric-value">5</div><div className="metric-sub">Imbalance handling methods</div></div></div><section className="panel model-panel"><div className="panel-heading"><div><h3>Experiment comparison</h3><p>Macro F1 scores on validation and held-out test pumps</p></div><span className="pill">15 EXPERIMENTS</span></div><div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={bestRows} margin={{top:10,right:12,left:-20,bottom:55}}><CartesianGrid strokeDasharray="3 3" stroke="#202b3b" vertical={false}/><XAxis dataKey={(d)=>`${labelize(d.strategy)} · ${labelize(d.model)}`} angle={-32} textAnchor="end" interval={0} height={75} tick={{fill:"#8290a5",fontSize:10}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{fill:"#8290a5",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111a28",border:"1px solid #263449",borderRadius:10,color:"#eaf0f8"}}/><Bar dataKey="val_macro_f1" name="Validation Macro F1" fill="#6c8cff" radius={[4,4,0,0]}/><Bar dataKey="test_macro_f1" name="Test Macro F1" fill="#39c7a0" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div><div className="table-scroll"><table><thead><tr><th>STRATEGY</th><th>MODEL</th><th>VALIDATION F1</th><th>TEST F1</th><th>TEST ACCURACY</th></tr></thead><tbody>{bestRows.map((r,i)=><tr key={i}><td>{labelize(r.strategy)}</td><td><b>{labelize(r.model)}</b></td><td>{fmt(r.val_macro_f1,4)}</td><td><b>{fmt(r.test_macro_f1,4)}</b></td><td>{fmt((r.test_accuracy||0)*100,2)}%</td></tr>)}</tbody></table></div></section><div className="note-panel"><ShieldCheck size={17}/><span>Model selection is based on validation Macro F1. The test set contains pump assets held out from training. The displayed test metrics are reported for evaluation and are not used to choose the deployed model.</span></div></motion.div>}
        </>}
        <footer><span>INTELLITWIN <i>•</i> INDUSTRIAL FAULT INTELLIGENCE</span><span>Historical dataset demonstration · Not connected to live plant sensors</span></footer>
      </div>
    </main>
  </div>
}
export default App;
'@ | Set-Content -Encoding UTF8 "frontend\src\App.jsx"

@'
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#e8edf5;background:#080d16;font-synthesis:none;text-rendering:optimizeLegibility}*{box-sizing:border-box}body{margin:0;min-width:320px;background:#080d16}button,input,select{font:inherit}button{cursor:pointer}.app-shell{min-height:100vh}.sidebar{width:248px;position:fixed;inset:0 auto 0 0;background:#0b111c;border-right:1px solid #1c2737;padding:25px 17px 16px;display:flex;flex-direction:column;z-index:10}.brand{display:flex;align-items:center;gap:11px;padding:0 8px 27px}.brand-mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(145deg,#1c9c91,#2563a5);display:grid;place-items:center;color:white;box-shadow:0 6px 24px #1c9c9140}.brand b{font-size:20px;letter-spacing:-1px}.brand b span{color:#55c9b1}.brand small{display:block;font-size:8px;letter-spacing:1.45px;color:#718096;margin-top:2px}.close-nav{display:none;margin-left:auto;background:none;border:0;color:#9aabc1}.workspace{border:1px solid #253247;border-radius:10px;padding:12px;display:flex;align-items:center;gap:9px;margin-bottom:31px}.workspace>div:nth-child(2){flex:1}.workspace b,.user-row b{font-size:11px;display:block}.workspace small,.user-row small{display:block;color:#7e8da2;font-size:10px;margin-top:4px}.workspace>svg{color:#718096}.workspace-dot,.green-dot,.live-dot{width:7px;height:7px;border-radius:50%;background:#3bd2a3;display:inline-block;box-shadow:0 0 10px #3bd2a366}.nav-label{font-size:9px;letter-spacing:1.5px;color:#64738a;font-weight:700;padding:0 11px 10px}.sidebar nav{display:grid;gap:5px}.nav-link{width:100%;border:0;background:transparent;color:#8493a8;border-radius:8px;padding:11px 12px;display:flex;align-items:center;gap:12px;text-align:left;font-size:12px;transition:.2s}.nav-link:hover{background:#141f2e;color:#dbe5f3}.nav-link.active{background:linear-gradient(90deg,#183b45,#142536);color:#6ce0c5;box-shadow:inset 2px 0 #50d1b2}.nav-link i{width:5px;height:5px;background:#54d6b5;border-radius:50%;margin-left:auto}.sidebar-bottom{margin-top:auto}.system-card{border:1px solid #253347;border-radius:10px;padding:13px;margin-bottom:15px;background:linear-gradient(145deg,#101e29,#101724)}.system-head{font-size:9px;letter-spacing:1px;color:#91a0b5;display:flex;align-items:center;gap:7px;margin-bottom:12px}.system-card>b{font-size:12px}.system-card>small{display:block;color:#8191a6;font-size:10px;margin-top:5px}.system-line{height:3px;background:#26394a;border-radius:5px;margin-top:12px}.system-line span{display:block;width:92%;height:100%;background:#3ac9a0;border-radius:5px}.user-row{border-top:1px solid #202b3a;margin-top:13px;padding:17px 4px 0;display:flex;align-items:center;gap:9px}.user-row>div:nth-child(2){flex:1}.avatar,.top-avatar{width:30px;height:30px;border-radius:9px;background:#233b52;color:#a9d9e9;display:grid;place-items:center;font-size:10px;font-weight:700}.main{margin-left:248px;min-height:100vh}.topbar{height:68px;border-bottom:1px solid #1c2737;display:flex;align-items:center;justify-content:space-between;padding:0 36px;background:#0a101a}.crumb{font-size:11px;color:#718096}.crumb span{padding:0 9px;color:#39485b}.crumb b{color:#d9e2ee;font-weight:600}.top-actions{display:flex;align-items:center;gap:17px}.data-status,.model-tag{font-size:9px;letter-spacing:1px;color:#7f91a7;display:flex;align-items:center;gap:8px}.icon-btn{background:none;border:0;color:#8c9bb0;position:relative;padding:5px}.icon-btn i{position:absolute;width:5px;height:5px;border-radius:50%;background:#f2ae53;right:3px;top:3px}.top-avatar{border-radius:50%;width:31px;height:31px}.content{max-width:1500px;margin:auto;padding:34px 37px 22px}.page-heading{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:23px;gap:20px}.eyebrow{font-size:9px;letter-spacing:1.6px;color:#5dcdb4;font-weight:700;display:flex;align-items:center;gap:7px;margin-bottom:9px}.page-heading h1{font-size:29px;letter-spacing:-1.2px;margin:0;font-weight:650}.page-heading p{color:#8290a4;font-size:12px;margin:8px 0 0}.secondary-btn,.primary-btn,.hero-btn{border:1px solid #304258;border-radius:8px;padding:10px 13px;background:#111d2b;color:#dce7f4;display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:600}.secondary-btn:hover{border-color:#5b8a9a}.primary-btn{background:linear-gradient(100deg,#168f85,#237ca3);border:0;color:white;justify-content:center;box-shadow:0 7px 22px #168f8530}.primary-btn:hover,.hero-btn:hover{filter:brightness(1.12)}.primary-btn:disabled{opacity:.6;cursor:wait}.full{width:100%}.hero{height:245px;border:1px solid #24404a;border-radius:14px;position:relative;overflow:hidden;background:radial-gradient(ellipse at 76% 43%,#17464c 0%,#122a35 27%,#101c2a 61%,#101a28 100%);padding:27px 31px;margin-bottom:19px}.hero-copy{position:relative;z-index:2;max-width:430px}.hero-kicker{font-size:9px;letter-spacing:1.5px;color:#77d9c4;display:flex;align-items:center;gap:8px}.hero-kicker span{width:6px;height:6px;background:#52d6b5;border-radius:50%;box-shadow:0 0 10px #52d6b5}.hero h2{font-size:31px;line-height:1.08;letter-spacing:-1.3px;margin:13px 0 8px}.hero h2 em{font-style:normal;color:#65d4bb}.hero-copy p{font-size:11px;color:#9eafbf;max-width:390px;line-height:1.6;margin:0 0 13px}.hero-btn{padding:9px 12px;background:#d9f8ef;color:#123c3d;border:0}.hero-visual{position:absolute;right:8%;top:0;width:370px;height:220px;display:grid;place-items:center;opacity:.9}.core{width:91px;height:91px;border-radius:50%;display:grid;place-items:center;color:#8af2db;background:radial-gradient(circle,#267e7b,#123d4b 68%);border:1px solid #56c8b5;box-shadow:0 0 50px #38d1b64d,0 0 0 15px #4dcbb30d;z-index:2}.orbit{position:absolute;border:1px solid #56c8b538;border-radius:50%;transform:rotate(-25deg)}.orbit-one{width:210px;height:110px}.orbit-two{width:295px;height:155px;transform:rotate(28deg)}.float-tag{position:absolute;z-index:3;background:#142d39;border:1px solid #3b716f;border-radius:6px;padding:7px 9px;font-size:8px;letter-spacing:.7px;color:#b8e7dc;display:flex;align-items:center;gap:6px}.tag-a{top:43px;right:20px}.tag-b{bottom:35px;left:15px}.visual-bars{position:absolute;bottom:10px;right:12px;height:40px;display:flex;align-items:end;gap:4px;opacity:.4}.visual-bars i{width:5px;background:#62d9c2;border-radius:3px}.hero-foot{position:absolute;bottom:13px;left:31px;right:31px;display:flex;gap:22px;font-size:8px;letter-spacing:.7px;color:#8fa5b7}.hero-foot span:first-child{color:#6cddc3}.hero-foot .green-dot{margin-right:6px;width:5px;height:5px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:19px}.metric-card,.panel{background:#0e1724;border:1px solid #202d3e;border-radius:11px}.metric-card{padding:15px 16px}.metric-top{display:flex;justify-content:space-between;align-items:center;color:#91a0b4;font-size:10px}.metric-icon,.panel-icon{width:32px;height:32px;border-radius:9px;display:grid;place-items:center}.metric-icon.blue{color:#79aaff;background:#1c2d48}.metric-icon.amber{color:#f2bc69;background:#3a3021}.metric-icon.green{color:#4ed6aa;background:#18372f}.metric-icon.violet{color:#b29aff;background:#302747}.metric-value{font-size:27px;font-weight:650;letter-spacing:-1px;margin:10px 0 4px}.metric-sub{font-size:9px;color:#75869c}.tiny-dot{width:5px;height:5px;display:inline-block;background:#45c8a2;border-radius:50%;margin-right:4px}.section-grid{display:grid;grid-template-columns:1.25fr 1fr;gap:14px;margin-bottom:26px}.panel{padding:17px}.panel-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.panel-heading h3,.section-heading h3{font-size:13px;margin:0;font-weight:620}.panel-heading p,.section-heading p{font-size:10px;color:#7e8da2;margin:5px 0 0}.pill{font-size:8px;letter-spacing:.8px;color:#8ca1b8;border:1px solid #2a3a4d;border-radius:20px;padding:5px 8px}.chart-wrap{height:205px;margin-top:13px}.panel-icon{color:#69cdbb;background:#17332f}.field-label{font-size:9px;letter-spacing:1px;color:#8293a9;display:block;margin:22px 0 8px}.select-wrap{height:40px;border:1px solid #2b3a4d;border-radius:7px;display:flex;align-items:center;padding:0 11px;background:#0a121e;color:#cbd6e4}.select-wrap select{width:100%;background:transparent;color:#dce5f0;border:0;outline:0;appearance:none;font-size:11px}.select-wrap option{background:#111c2a}.select-wrap svg{pointer-events:none;color:#8191a6}.quick-sensors{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:13px 0}.quick-sensors>div{background:#121e2d;border:1px solid #233246;border-radius:7px;padding:9px}.quick-sensors span{display:block;color:#7e8ea3;font-size:9px;margin-bottom:5px}.quick-sensors b{font-size:12px}.quick-sensors small{font-size:8px;color:#7f91a7;font-weight:400}.helper-note{display:flex;align-items:center;gap:6px;color:#71859b;font-size:9px;margin:10px 0 0}.helper-note svg{color:#54c9ad;flex-shrink:0}.section-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:11px}.text-btn{background:none;border:0;color:#62cbb6;font-size:10px;display:flex;align-items:center;gap:5px}.table-panel{border:1px solid #202d3e;background:#0e1724;border-radius:10px;overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}th{font-size:8px;letter-spacing:.9px;color:#73839a;font-weight:600;padding:13px 16px;border-bottom:1px solid #202d3e}td{font-size:10px;color:#a8b5c6;padding:13px 16px;border-bottom:1px solid #1c2938}tbody tr:last-child td{border-bottom:0}tbody tr{cursor:pointer}tbody tr:hover{background:#121f2d}.asset-id{color:#d9e5f2}.fault-label{font-size:9px;color:var(--fault);background:color-mix(in srgb,var(--fault) 12%,transparent);border:1px solid color-mix(in srgb,var(--fault) 28%,transparent);padding:5px 8px;border-radius:5px}.searchbox{display:flex;align-items:center;gap:8px;background:#0e1724;border:1px solid #29384b;border-radius:8px;padding:9px 12px;color:#7e8da2}.searchbox input{background:transparent;border:0;outline:0;color:#dce5ef;font-size:11px;width:160px}.fleet-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.fleet-card{border:1px solid #233145;background:#0e1724;border-radius:10px;text-align:left;padding:15px;color:#e4ebf4;transition:.2s}.fleet-card:hover,.fleet-card.chosen{border-color:#3a8e85;background:#101e2a}.fleet-card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:13px}.pump-icon{width:34px;height:34px;border-radius:9px;background:#1c2c40;color:#78b8d8;display:grid;place-items:center}.risk-chip{font-size:8px;border-radius:20px;padding:5px 8px;border:1px solid}.risk-chip.green{color:#50d2a7;background:#15342d;border-color:#285b4d}.risk-chip.amber{color:#f0bc6c;background:#382d1d;border-color:#604a2d}.risk-chip.red{color:#ff8290;background:#3d222b;border-color:#693640}.fleet-card>b{font-size:13px;display:block}.fleet-card>small{display:block;color:#8190a4;font-size:9px;margin-top:4px}.fleet-readings{display:flex;gap:25px;margin:17px 0 13px}.fleet-readings span{font-size:8px;color:#77889d;letter-spacing:.5px}.fleet-readings strong{display:block;color:#d8e2ef;font-size:11px;margin-top:5px;letter-spacing:0}.fleet-bottom{border-top:1px solid #233044;padding-top:12px;display:flex;align-items:center;justify-content:space-between}.fleet-bottom>span:last-child{font-size:9px;color:#8796a9}.selected-action{margin-top:15px;padding:14px 16px;border:1px solid #284c4d;background:#102329;border-radius:9px;display:flex;align-items:center;justify-content:space-between}.selected-action b,.selected-action span{display:block}.selected-action b{font-size:11px}.selected-action span{font-size:9px;color:#8196a7;margin-top:4px}.diagnosis-layout{display:grid;grid-template-columns:1fr 1fr;gap:15px}.diagnosis-form,.result-panel{min-height:445px}.diagnosis-form>.panel-heading{margin-bottom:10px}.sensor-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:15px}.sensor-box{background:#121e2b;border:1px solid #233246;border-radius:8px;padding:11px;display:grid;grid-template-columns:20px 1fr;align-items:center;gap:4px 7px;color:#6faeae}.sensor-box span{font-size:9px;color:#8b9aaf}.sensor-box b{grid-column:2;color:#e2eaf3;font-size:12px}.sensor-box small{font-size:9px;color:#8392a7;font-weight:400}.diagnose-button{margin-top:17px;padding:12px}.result-placeholder{margin-top:17px;background:#111d2a;border:1px solid #26374b;border-radius:10px;padding:19px;min-height:345px}.result-overline{font-size:8px;letter-spacing:1.3px;color:#8395a9}.result-name{font-size:27px;font-weight:700;letter-spacing:-.8px;margin:9px 0 10px}.confidence-row{display:flex;justify-content:space-between;font-size:10px;color:#a5b2c3;margin:20px 0 8px}.confidence-row b{color:#eef4fa;font-size:13px}.confidence-track,.prob-row>div{height:6px;background:#263449;border-radius:10px;overflow:hidden}.confidence-track i,.prob-row i{display:block;height:100%;border-radius:10px}.prob-list{display:grid;gap:13px;margin-top:21px}.prob-row{display:grid;grid-template-columns:100px 1fr 42px;align-items:center;gap:9px}.prob-row span{font-size:9px;color:#aab7c7}.prob-row b{text-align:right;font-size:9px}.prob-row>div{height:5px}.result-disclaimer{border-top:1px solid #2a394a;margin-top:18px;padding-top:12px;font-size:9px;color:#8796a9;line-height:1.5}.empty-result{min-height:310px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;max-width:280px;margin:auto}.empty-icon{width:58px;height:58px;border-radius:18px;background:#1b3040;color:#62cbb5;display:grid;place-items:center;margin-bottom:15px}.empty-result b{font-size:13px}.empty-result span{font-size:10px;color:#8393a8;line-height:1.6;margin:7px 0 16px}.model-panel{margin-top:17px}.chart-wrap.tall{height:330px;margin-top:20px}.table-scroll{overflow:auto;margin-top:15px}.compact{margin-top:18px}.note-panel{display:flex;gap:10px;align-items:flex-start;padding:13px 15px;background:#102229;border:1px solid #284b4c;border-radius:8px;margin-top:14px;color:#9bb4ba;font-size:10px;line-height:1.55}.note-panel svg{color:#5dd4b7;flex-shrink:0}footer{display:flex;justify-content:space-between;gap:15px;border-top:1px solid #1d2939;margin-top:28px;padding:16px 0 0;color:#5f7187;font-size:8px;letter-spacing:.7px}footer i{color:#3ec7a2;font-style:normal;margin:0 5px}.loading{min-height:65vh;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;color:#dce6f1}.loading span{font-size:10px;color:#8090a5}.loader{width:29px;height:29px;border:2px solid #26384b;border-top-color:#50d1b2;border-radius:50%;animation:spin .8s linear infinite;margin-bottom:6px}.spin{animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.error-banner{display:flex;align-items:center;gap:9px;background:#3b2028;border:1px solid #713846;color:#ffabb4;padding:11px 13px;border-radius:8px;font-size:10px;margin-bottom:16px}.error-banner button{margin-left:auto;background:none;border:0;color:inherit}.mobile-menu{display:none;background:none;border:0;color:#a8b8ca}.scrim{display:none}
@media(max-width:1150px){.content{padding:28px 24px}.hero-visual{right:1%;opacity:.65}.fleet-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.section-grid{grid-template-columns:1fr 1fr}}
@media(max-width:850px){.sidebar{transform:translateX(-100%);transition:transform .25s}.sidebar.show{transform:translateX(0)}.close-nav{display:block}.scrim{display:block;position:fixed;inset:0;background:#0009;z-index:9}.main{margin-left:0}.mobile-menu{display:block}.topbar{padding:0 20px}.content{padding:25px 20px}.hero-visual{right:-70px;opacity:.35}.section-grid,.diagnosis-layout{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}.diagnosis-form,.result-panel{min-height:unset}}
@media(max-width:560px){.content{padding:20px 14px}.topbar{height:58px;padding:0 14px}.data-status{display:none}.page-heading{align-items:flex-start;flex-direction:column}.page-heading h1{font-size:25px}.hero{height:265px;padding:22px}.hero h2{font-size:27px}.hero-visual{right:-145px;opacity:.24}.hero-foot{left:22px;right:10px;gap:12px;font-size:7px}.metric-grid{gap:8px}.metric-card{padding:12px}.metric-value{font-size:23px}.section-grid{gap:10px}.fleet-grid{grid-template-columns:1fr}.selected-action{align-items:flex-start;gap:12px;flex-direction:column}.searchbox{width:100%}.searchbox input{width:100%}footer{flex-direction:column}.top-actions{gap:9px}.diagnosis-layout{gap:10px}.prob-row{grid-template-columns:88px 1fr 38px}}
'@ | Set-Content -Encoding UTF8 "frontend\src\style.css"

Write-Host "[3/5] Installing backend dependencies..." -ForegroundColor Cyan
python -m pip install fastapi "uvicorn[standard]"

Write-Host "[4/5] Installing frontend dependencies..." -ForegroundColor Cyan
Push-Location "frontend"
npm install
Pop-Location

Write-Host "[5/5] Starting IntelliTwin..." -ForegroundColor Cyan
$BackendCommand = "Set-Location '$Root'; .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
if (!(Test-Path ".venv\Scripts\python.exe")) {
    $BackendCommand = "Set-Location '$Root'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
}
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCommand
Start-Sleep -Seconds 3
$FrontendCommand = "Set-Location '$Root\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCommand
Start-Sleep -Seconds 4
Start-Process "http://127.0.0.1:5173"

Write-Host "`nIntelliTwin launch requested." -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Backend docs: http://127.0.0.1:8000/docs"
Write-Host "Keep both server windows open while demonstrating." -ForegroundColor Yellow
