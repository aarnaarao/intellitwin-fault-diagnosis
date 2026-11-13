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
      setDiagnosis(data); setPage("Fault Diagnosis");
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
          {machines.some(m=>m.predicted_fault!=="Normal")&&<section className="panel alert-panel"><div className="panel-heading"><div><h3>Priority fault alerts</h3><p>Assets classified with a non-normal condition in the latest historical reading</p></div><span className="pill">{machines.filter(m=>m.predicted_fault!=="Normal").length} ACTIVE</span></div><div className="alert-list">{machines.filter(m=>m.predicted_fault!=="Normal").sort((a,b)=>b.confidence-a.confidence).map(m=><button className="alert-item" key={m.asset_id} onClick={()=>{setSelected(m.asset_id);setPage("Fault Diagnosis");setDiagnosis(null);}}><span className={`alert-symbol ${m.predicted_fault==="Shutdown"?"shutdown":"overload"}`}><AlertTriangle size={17}/></span><span className="alert-copy"><b>{m.asset_id} · {labelize(m.predicted_fault)}</b><small>{m.pump_type} · latest historical reading</small></span><span className="alert-confidence">{m.confidence_percent}%<small>confidence</small></span><ArrowUpRight size={16}/></button>)}</div></section>}
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
