#!/usr/bin/env python3
"""Build dashboard HTML v3 — complete UI/UX redesign."""

import json, csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "model_ready_v2"
MODELS = BASE / "models"
OUT = Path("/private/tmp/claude-501/-Users-filippoferrari-Documents-UnB-TCC/"
           "d724456e-daa0-4369-96dd-574a3a370146/scratchpad/dashboard.html")

def read_json(p):
    return json.load(open(p)) if p.exists() else {}

def read_csv_rows(p):
    return list(csv.DictReader(open(p))) if p.exists() else []

D = {}
for f in sorted(DATA.glob("*.json")):
    D[f.stem] = read_json(f)

csv_map = {
    "eda_spearman": DATA / "eda_spearman_matrix.csv",
    "eda_pearson": DATA / "eda_pearson_matrix.csv",
    "eda_regiao": DATA / "eda_por_regiao.csv",
    "eda_lag_otimo": DATA / "eda_lag_otimo.csv",
    "eda_lags_bio": DATA / "eda_lags_biologicos.csv",
    "metrics_uf": MODELS / "regression_5yr" / "metrics_por_uf.csv",
    "metrics_regiao": MODELS / "regression_5yr" / "metrics_por_regiao.csv",
    "metrics_uf_v2": MODELS / "regression_v2" / "metrics_por_uf_v2.csv",
    "shap_p1": MODELS / "shap_5yr" / "shap_feature_importance.csv",
    "shap_v2": MODELS / "shap_v2" / "shap_feature_importance_v2.csv",
}
for key, path in csv_map.items():
    D[key] = read_csv_rows(path)

data_json = json.dumps(D, ensure_ascii=False, default=str)

html_template = r"""<title>Dengue × Clima</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ff-b:'IBM Plex Sans',system-ui,sans-serif;
  --ff-m:'IBM Plex Mono','Menlo',monospace;
  --hover-bg:rgba(0,0,0,.04);
  --bg:#F5F7FA;--card:#FFFFFF;--card-alt:#EFF2F7;
  --border:#DDE2EB;--border-dim:#EDF0F5;
  --tx:#1A1F2B;--tx2:#4B5563;--tx3:#9CA3AF;
  --c1:#2a78d6;--c1-bg:rgba(42,120,214,.07);
  --c2:#eb6834;--c2-bg:rgba(235,104,52,.07);
  --ok:#0ca30c;--ok-bg:rgba(12,163,12,.07);
  --bad:#d03b3b;--bad-bg:rgba(208,59,59,.07);
  --dup:#006300;
  --grid:rgba(0,0,0,.06);--axis:#D1D5DB;
  --sh:0 1px 2px rgba(0,0,0,.04);--r:8px;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0E1117;--card:#161B25;--card-alt:#1C2130;
  --border:#262D3D;--border-dim:#1E2535;
  --tx:#E5E7EB;--tx2:#9CA3AF;--tx3:#6B7280;
  --c1:#3987e5;--c1-bg:rgba(57,135,229,.10);
  --c2:#d95926;--c2-bg:rgba(217,89,38,.10);
  --ok-bg:rgba(12,163,12,.10);--bad-bg:rgba(208,59,59,.10);
  --dup:#0ca30c;
  --grid:rgba(255,255,255,.06);--axis:#374151;
  --sh:0 1px 2px rgba(0,0,0,.15);--hover-bg:rgba(255,255,255,.06);
}}
:root[data-theme="dark"]{
  --bg:#0E1117;--card:#161B25;--card-alt:#1C2130;
  --border:#262D3D;--border-dim:#1E2535;
  --tx:#E5E7EB;--tx2:#9CA3AF;--tx3:#6B7280;
  --c1:#3987e5;--c1-bg:rgba(57,135,229,.10);
  --c2:#d95926;--c2-bg:rgba(217,89,38,.10);
  --ok-bg:rgba(12,163,12,.10);--bad-bg:rgba(208,59,59,.10);
  --dup:#0ca30c;
  --grid:rgba(255,255,255,.06);--axis:#374151;
  --sh:0 1px 2px rgba(0,0,0,.15);--hover-bg:rgba(255,255,255,.06);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--ff-b);background:var(--bg);color:var(--tx);line-height:1.6;font-size:16px}
h1,h2{font-family:var(--ff-b);font-weight:700}
h2{font-size:1.6rem;text-wrap:balance;color:var(--tx);letter-spacing:-.02em}
h3{font:600 .92rem/1.4 var(--ff-b);color:var(--tx)}

/* Header */
header{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--border)}
header::before{content:'';display:block;height:3px;background:linear-gradient(90deg,var(--c1) 0%,var(--c2) 100%)}
.hdr{max-width:960px;margin:0 auto;padding:0 24px}
.brand{display:flex;align-items:center;justify-content:space-between;padding:12px 0 0}
.brand-name{font:700 1.25rem var(--ff-b);color:var(--tx);letter-spacing:-.02em}
.brand-right{display:flex;align-items:center;gap:10px}
.brand-sub{font:500 .72rem var(--ff-m);color:var(--tx3)}
.theme-btn{background:var(--hover-bg);border:1px solid var(--border);border-radius:6px;padding:4px 12px;cursor:pointer;color:var(--tx3);font:500 .72rem var(--ff-m);transition:all .15s}
.theme-btn:hover{color:var(--tx);border-color:var(--tx3);background:var(--card-alt)}
.tabs{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;padding:8px 0 6px}
.tabs::-webkit-scrollbar{display:none}
.tab{padding:7px 14px;color:var(--tx3);font:600 .84rem var(--ff-b);border:none;background:none;border-radius:6px;cursor:pointer;white-space:nowrap;transition:all .12s}
.tab:hover{color:var(--tx2);background:var(--hover-bg)}
.tab.on{color:var(--c1);background:var(--c1-bg)}

/* Main */
main{max-width:960px;margin:0 auto;padding:28px 24px 48px}
.sec{display:none}.sec.on{display:block}
.sec-head{margin-bottom:20px}
.sec-head p{color:var(--tx2);font-size:.94rem;margin-top:6px;max-width:65ch;line-height:1.65}

/* KPI grid */
.kpi-grid{display:grid;gap:10px;margin-bottom:20px}
.k4{grid-template-columns:repeat(4,1fr)}
.k3{grid-template-columns:repeat(3,1fr)}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px}
.kpi-v{font:600 1.6rem/1.2 var(--ff-b);color:var(--tx)}
.kpi-l{font-size:.72rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.kpi-d{font:400 .8rem var(--ff-m);margin-top:3px}
.kpi-d.up{color:var(--dup)}.kpi-d.dn{color:var(--bad)}

/* Callout */
.co{border-left:3px solid var(--c1);padding:12px 16px;margin-bottom:14px;background:var(--c1-bg);border-radius:0 6px 6px 0}
.co.im{border-color:var(--c2);background:var(--c2-bg)}
.co.ok{border-color:var(--ok);background:var(--ok-bg)}
.co.bd{border-color:var(--bad);background:var(--bad-bg)}
.co h3{margin-bottom:3px}
.co p{font-size:.9rem;color:var(--tx2);line-height:1.6}

/* Note */
.note{font-size:.9rem;color:var(--tx2);padding:12px 16px;background:var(--card-alt);border-radius:6px;margin-bottom:14px;line-height:1.65}
.note strong{color:var(--tx)}

/* Chart */
.cw{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px;margin-bottom:12px;box-shadow:var(--sh)}
.cw h3{font-size:.88rem;color:var(--tx2);margin-bottom:10px;font-family:var(--ff-b);font-weight:600}
.cw .cc{position:relative;width:100%;height:320px}
.cw.tall .cc{height:480px}
.cw canvas{display:block}
.chart-err{color:var(--bad);font-size:.85rem;padding:16px;text-align:center;min-height:100px;display:flex;align-items:center;justify-content:center}

/* Table */
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:12px;box-shadow:var(--sh)}
.tw .th{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--border)}
.tw .th h3{font:600 .82rem/1 var(--ff-b);color:var(--tx);margin:0}
.tw input{font:.78rem var(--ff-b);padding:5px 10px;border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--tx);width:170px}
table{width:100%;border-collapse:collapse;font-size:.84rem;font-variant-numeric:tabular-nums}
th{text-align:left;padding:8px 12px;font:600 .72rem/1 var(--ff-b);text-transform:uppercase;letter-spacing:.04em;color:var(--tx3);border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
th:hover{color:var(--tx2)}
td{padding:7px 12px;border-bottom:1px solid var(--border-dim);font:.82rem var(--ff-m);color:var(--tx)}
tr:hover td{background:var(--card-alt)}
.bg{display:inline-block;padding:1px 6px;border-radius:3px;font:600 .66rem var(--ff-b);text-transform:uppercase;letter-spacing:.03em}
.bg.si{background:var(--c1-bg);color:var(--c1)}.bg.im{background:var(--c2-bg);color:var(--c2)}

/* Threshold cards */
.thr-g{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.thr{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px;text-align:center;transition:box-shadow .12s}
.thr:hover{box-shadow:0 4px 12px rgba(0,0,0,.06)}
.thr .nm{font:.73rem var(--ff-m);color:var(--c2);margin-bottom:5px;word-break:break-all}
.thr .tv{font:500 1.3rem var(--ff-b);color:var(--tx)}
.thr .dr{font-size:.7rem;color:var(--tx3);margin-top:3px}

/* Stages grid */
.st-g{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:16px}
.st{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px}
.st h3{color:var(--c1);margin-bottom:4px}
.st p{font-size:.84rem;color:var(--tx2);line-height:1.55}
.st .meta{font:.76rem var(--ff-m);color:var(--tx3);margin-top:4px}

/* Answer block */
.answer{border:2px solid var(--c1);border-radius:var(--r);padding:20px 24px;margin-bottom:20px;background:var(--c1-bg)}
.answer-q{font:500 .78rem var(--ff-b);text-transform:uppercase;letter-spacing:.06em;color:var(--c1);margin-bottom:6px}
.answer-a{font:400 1.15rem/1.55 var(--ff-b);color:var(--tx)}
.answer-a strong{color:var(--c1)}
.answer-detail{font:.85rem var(--ff-b);color:var(--tx2);margin-top:8px;line-height:1.6}
.answer-badge{display:inline-block;background:var(--card);border:1px solid var(--border);padding:3px 10px;border-radius:4px;font:.72rem var(--ff-m);color:var(--tx3);margin-top:10px}

/* Loading */
.loading{display:flex;align-items:center;justify-content:center;padding:80px 20px;color:var(--tx3);font:500 .9rem var(--ff-b);gap:10px}
.loading::before{content:'';width:20px;height:20px;border:2.5px solid var(--border);border-top-color:var(--c1);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Responsive */
@media(max-width:768px){.k4{grid-template-columns:repeat(2,1fr)}.k3{grid-template-columns:repeat(2,1fr)}.st-g{grid-template-columns:1fr}}
@media(max-width:480px){main{padding:16px 12px}.k4,.k3{grid-template-columns:1fr}.tab{padding:8px 10px;font-size:.75rem}}
</style>

<header>
  <div class="hdr">
    <div class="brand">
      <span class="brand-name">Dengue × Clima</span>
      <div class="brand-right">
        <span class="brand-sub">XGBoost · SINAN + INMET</span>
        <button class="theme-btn" onclick="toggleTheme()">tema</button>
      </div>
    </div>
    <nav class="tabs">
      <button class="tab on" data-s="overview" onclick="go('overview')">Resumo</button>
      <button class="tab" data-s="blindness" onclick="go('blindness')">Data Blindness</button>
      <button class="tab" data-s="eda" onclick="go('eda')">EDA</button>
      <button class="tab" data-s="regression" onclick="go('regression')">Regressão</button>
      <button class="tab" data-s="classification" onclick="go('classification')">Classificação</button>
      <button class="tab" data-s="shap" onclick="go('shap')">SHAP</button>
      <button class="tab" data-s="walkforward" onclick="go('walkforward')">Walk-Forward</button>
      <button class="tab" data-s="methodology" onclick="go('methodology')">Método</button>
    </nav>
  </div>
</header>

<main>
  <div class="sec on" id="s-overview"></div>
  <div class="sec" id="s-blindness"></div>
  <div class="sec" id="s-eda"></div>
  <div class="sec" id="s-regression"></div>
  <div class="sec" id="s-classification"></div>
  <div class="sec" id="s-shap"></div>
  <div class="sec" id="s-walkforward"></div>
  <div class="sec" id="s-methodology"></div>
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D = __DATA__;
const R = {};
let chartOk=false;
try{if(typeof Chart!=='undefined'){if(Chart.registerables)Chart.register(...Chart.registerables);chartOk=true}}catch(e){console.error('Chart init failed',e)}

try{const s=localStorage.getItem('t');if(s)document.documentElement.setAttribute('data-theme',s)}catch(e){}

function isDark(){
  const dt=document.documentElement.getAttribute('data-theme');
  if(dt==='dark')return true;if(dt==='light')return false;
  return window.matchMedia('(prefers-color-scheme:dark)').matches;
}
function c1(){return isDark()?'#3987e5':'#2a78d6'}
function c2(){return isDark()?'#d95926':'#eb6834'}

function go(id){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  const sec=document.getElementById('s-'+id);
  sec.classList.add('on');
  document.querySelector('[data-s="'+id+'"]').classList.add('on');
  window.scrollTo(0,0);
  if(!R[id]){
    sec.innerHTML='<div class="loading">Carregando</div>';
    requestAnimationFrame(()=>requestAnimationFrame(()=>{sec.innerHTML='';P[id]();R[id]=1}));
  }
}
function toggleTheme(){
  const r=document.documentElement,c=r.getAttribute('data-theme');
  const n=c==='dark'?'light':'dark';
  r.setAttribute('data-theme',n);
  try{localStorage.setItem('t',n)}catch(e){}
  const act=document.querySelector('.tab.on')?.dataset.s;
  if(act){
    const sec=document.getElementById('s-'+act);
    if(chartOk)sec.querySelectorAll('canvas').forEach(cv=>{try{const ch=Chart.getChart(cv);if(ch)ch.destroy()}catch(e){}});
    sec.innerHTML='';R[act]=0;P[act]();R[act]=1;
  }
}

const fmt=(v,d=4)=>{const n=parseFloat(v);return isNaN(n)?String(v||'—'):n.toFixed(d)};
const f2=v=>fmt(v,2);
const pv=v=>{const n=parseFloat(v);return isNaN(n)?'—':n<0.001?n.toExponential(2):n.toFixed(4)};
const dc=v=>parseFloat(v)>0?'up':parseFloat(v)<0?'dn':'';

function el(t,a,...ch){
  const e=document.createElement(t);
  if(a)Object.entries(a).forEach(([k,v])=>{
    if(k==='cls')e.className=v;
    else if(k==='sty'&&typeof v==='object')Object.assign(e.style,v);
    else if(k.startsWith('on'))e.addEventListener(k.slice(2).toLowerCase(),v);
    else e.setAttribute(k,v);
  });
  ch.flat().forEach(c=>{if(c!=null)e.appendChild(typeof c==='string'?document.createTextNode(c):c)});
  return e;
}

function mkChart(box,cfg){
  const tall=cfg.aspect&&cfg.aspect<2;
  const w=el('div',{cls:tall?'cw tall':'cw'});
  if(cfg.title)w.appendChild(el('h3',null,cfg.title));
  if(!chartOk){w.appendChild(el('div',{cls:'chart-err'},'Chart.js não disponível'));box.appendChild(w);return null}
  const cc=el('div',{cls:'cc'});
  const canvas=el('canvas');
  cc.appendChild(canvas);w.appendChild(cc);box.appendChild(w);
  try{
    const dark=isDark();
    const grid=dark?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)';
    const tick=dark?'#9CA3AF':'#6B7280';
    const axisC=dark?'#374151':'#D1D5DB';
    const ttBg=dark?'#1C2130':'#FFFFFF';
    const ttTx=dark?'#E5E7EB':'#1A1F2B';
    const ttTx2=dark?'#9CA3AF':'#4B5563';
    const ttBr=dark?'#262D3D':'#DDE2EB';
    const scales={};
    if(cfg.type==='bar'||cfg.type==='line'){
      const isH=cfg.opts?.indexAxis==='y';
      scales.x={grid:{color:grid},ticks:{color:tick,font:{family:"'IBM Plex Mono'",size:10}},border:{color:axisC},...(isH?(cfg.yAxis||{}):(cfg.xAxis||{}))};
      scales.y={grid:{color:grid},ticks:{color:tick,font:{family:"'IBM Plex Mono'",size:10}},border:{color:axisC},...(isH?(cfg.xAxis||{}):(cfg.yAxis||{}))};
    }
    const ch=new Chart(canvas,{
      type:cfg.type,data:cfg.data,
      options:{
        responsive:true,maintainAspectRatio:false,
        layout:{padding:{top:4,right:4}},
        plugins:{
          legend:{display:cfg.legend!==false,labels:{color:tick,font:{family:"'IBM Plex Sans'",size:12},usePointStyle:true,pointStyle:'circle',padding:14}},
          tooltip:{backgroundColor:ttBg,titleColor:ttTx,bodyColor:ttTx2,borderColor:ttBr,borderWidth:1,
            padding:{x:12,y:8},cornerRadius:6,
            titleFont:{family:"'IBM Plex Sans'",weight:'600',size:12},bodyFont:{family:"'IBM Plex Mono'",size:12},
            ...(cfg.tooltipOpts||{})},
          ...(cfg.plugins||{})
        },
        scales:Object.keys(scales).length?scales:undefined,
        ...(cfg.opts||{})
      }
    });
    return ch;
  }catch(err){w.appendChild(el('div',{cls:'chart-err'},'Erro: '+err.message));return null}
}

function mkTbl(box,cfg){
  const w=el('div',{cls:'tw'});
  const h=el('div',{cls:'th'},el('h3',null,cfg.title+(cfg.rows.length?' ('+cfg.rows.length+')':'')),
    cfg.search!==false?el('input',{placeholder:'Buscar...',onInput:e=>filt(tb,e.target.value)}):null);
  w.appendChild(h);
  const t=el('table');
  t.appendChild(el('thead',null,el('tr',null,...cfg.hdr.map((h,i)=>el('th',{onClick:()=>srt(tb,i)},h)))));
  const tb=el('tbody');
  cfg.rows.forEach(row=>{
    tb.appendChild(el('tr',null,...row.map(c=>{const td=el('td');if(c&&c.nodeType)td.appendChild(c);else td.textContent=c??'';return td})));
  });
  t.appendChild(tb);w.appendChild(t);box.appendChild(w);
}
function filt(tb,q){const l=q.toLowerCase();Array.from(tb.rows).forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(l)})}
function srt(tb,i){
  const rows=Array.from(tb.rows),d=tb.dataset.sd==='a'?'d':'a';tb.dataset.sd=d;
  rows.sort((a,b)=>{let va=a.cells[i]?.textContent?.trim()??'',vb=b.cells[i]?.textContent?.trim()??'';
    const na=parseFloat(va),nb=parseFloat(vb);
    if(!isNaN(na)&&!isNaN(nb))return d==='a'?na-nb:nb-na;
    return d==='a'?va.localeCompare(vb):vb.localeCompare(va);
  });rows.forEach(r=>tb.appendChild(r));
}

const P={

overview(){
  const e=document.getElementById('s-overview');
  const reg=D['13_regression_v2_report']?.modelos||{};
  const bl=D['14_blindness_report']?.cenarios||{};
  const sh=D['16_shap_v2_report']||{};
  const wf=D['17_walkforward_v2_report']||{};
  const cls=D['15_classification_v2_report']||{};
  const r2b=reg.B_inmet_bruto?.R2_log||0,r2a=reg.A_sinan_only?.R2_log||0;
  const delta=r2b-r2a;
  const blD=bl['INMET-only']?.delta_R2_log||0;

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Variáveis climáticas melhoram a previsão de dengue?'),
    el('p',null,'18 experimentos XGBoost treinados sobre 1.224.164 registros reais de dengue (SINAN) cruzados com dados meteorológicos (INMET) em 4.406 municípios brasileiros ao longo de 6 anos.')
  ));

  e.appendChild(el('div',{cls:'answer'},
    el('div',{cls:'answer-q'},'Resultado'),
    el('div',{cls:'answer-a'},
      'A relação existe mas é ',el('strong',null,'marginal'),' com dados SINAN completos — ',
      'ΔR² = +'+fmt(delta)+', p < 10⁻¹⁸. O valor real do INMET aparece na ',
      el('strong',null,'ausência de dados epidemiológicos'),': quando o SINAN está indisponível, ',
      'o clima é o único sinal preditivo (ΔR² = +'+fmt(blD)+').'
    ),
    el('div',{cls:'answer-detail'},
      'O modelo identificou limiares biológicos reais: temperatura > 25.7°C com defasagem de 8 semanas — ',
      'consistente com o ciclo completo do Aedes aegypti (ovo → larva → mosquito → picada → sintomas → notificação).'
    ),
    el('span',{cls:'answer-badge'},'1.224.164 registros · 4.406 municípios · 6 anos · XGBoost')
  ));

  const kg=el('div',{cls:'kpi-grid k4'});
  [[fmt(r2b),'R² log (melhor modelo)','+'+fmt(delta)+' vs SINAN-only','up'],
   ['+'+fmt(blD),'Data Blindness','INMET-only: único sinal','up'],
   [String(sh.climate_in_top_20||0)+'/20','SHAP top-20 clima','rank '+(sh.best_climate_rank||'?'),''],
   [String(wf.folds_inmet_better||0)+'/5','Walk-forward folds','Δ médio: '+fmt(wf.delta_mean||0),dc(wf.delta_mean)]
  ].forEach(([v,l,s,c])=>kg.appendChild(el('div',{cls:'kpi'},el('div',{cls:'kpi-v'},v),el('div',{cls:'kpi-l'},l),el('div',{cls:'kpi-d '+c},s))));
  e.appendChild(kg);

  [['Data Blindness é o argumento central',
    'Secretarias enfrentam atrasos de 4-8 semanas no SINAN. No cenário Blind-4w (atraso realista), o ganho é significativo (p=0.024). Quando o SINAN está indisponível, o INMET é o único sinal (ΔR² = +0.1182).','im'],
   ['Assinatura biológica confirmada',
    'O XGBoost descobriu que temperatura > 25.7°C com lag de 8 semanas aumenta o risco — correspondendo ao ciclo ovo → larva → mosquito → picada → sintomas → notificação documentado na literatura.','ok'],
   ['Classificação supera regressão para políticas públicas',
    'Saber se haverá surto (4 classes de risco) é mais útil que prever o número exato de casos. O clima funciona como gatilho de categoria — modelo C (INMET enriquecido, F1='+fmt(cls.modelos?.['C: INMET enriquecido']?.f1_macro||0)+') vence na classificação.','im'],
   ['Modelos históricos vs. mudanças climáticas',
    'Walk-forward mostrou queda em 2025/2026 ao usar dados climáticos — anomalias extremas recentes (El Niño atípico) estão quebrando padrões históricos.','bd']
  ].forEach(([t,p,c])=>e.appendChild(el('div',{cls:'co '+c},el('h3',null,t),el('p',null,p))));
},

blindness(){
  const e=document.getElementById('s-blindness');
  const bl=D['14_blindness_report']?.cenarios||{};
  const entries=Object.entries(bl);

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Data Blindness — Simulação de Atraso SINAN'),
    el('p',null,'O que acontece quando dados epidemiológicos estão atrasados ou indisponíveis? Features SINAN recentes são mascaradas (zeradas) e comparamos o desempenho com e sem INMET.')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'Por que importa: '),
    'Municípios pequenos demoram 4-8 semanas para consolidar fichas de notificação. O INMET (estações automáticas) transmite em tempo real. Se o clima prevê dengue nessas janelas, funciona como radar antecipado.'
  ));

  e.appendChild(el('div',{cls:'co im'},
    el('h3',null,'INMET é o único sinal quando SINAN está indisponível'),
    el('p',null,'No cenário INMET-only, o modelo com INMET atinge R²=0.1265 vs 0.0083 sem — diferença de +0.1182. No Blind-4w (atraso prático), melhora significativamente (p=0.024). No Blind-8w (atraso severo), o INMET atrapalha — possivelmente overfitting.')
  ));

  if(entries.length){
    const labels=entries.map(([k])=>k);
    const com=entries.map(([,v])=>v.com_inmet?.R2_log||0);
    const sem=entries.map(([,v])=>v.sem_inmet?.R2_log||0);

    mkChart(e,{title:'R² log — Com vs Sem INMET por cenário',type:'bar',
      data:{labels,datasets:[
        {label:'Com INMET',data:com,backgroundColor:c2(),borderRadius:4,maxBarThickness:28},
        {label:'Sem INMET',data:sem,backgroundColor:c1(),borderRadius:4,maxBarThickness:28}
      ]},
      tooltipOpts:{callbacks:{label:c=>c.dataset.label+': '+c.raw.toFixed(4)}}
    });

    const deltas=entries.map(([,v])=>v.delta_R2_log||0);
    mkChart(e,{title:'ΔR² — Contribuição do INMET por cenário',type:'bar',aspect:2.5,
      data:{labels,datasets:[{label:'ΔR²',data:deltas,
        backgroundColor:deltas.map(d=>d>0?'#0ca30c':'#d03b3b'),borderRadius:4,maxBarThickness:28}]},
      plugins:{legend:{display:false}},opts:{indexAxis:'y'},
      tooltipOpts:{callbacks:{label:c=>'Δ: '+(c.raw>0?'+':'')+c.raw.toFixed(4)}}
    });

    mkTbl(e,{title:'Detalhes por cenário',search:false,
      hdr:['Cenário','R² com INMET','R² sem INMET','ΔR²','p-value','Ajuda?'],
      rows:entries.map(([name,d])=>[name,fmt(d.com_inmet?.R2_log),fmt(d.sem_inmet?.R2_log),
        (d.delta_R2_log>0?'+':'')+fmt(d.delta_R2_log),
        d.ttest?.p_value!=null?pv(d.ttest.p_value):'—',
        d.inmet_ajuda?'✓ Sim':'✗ Não'])
    });
  }
},

eda(){
  const e=document.getElementById('s-eda');
  const eda=D['02_eda_report']||{};
  const lags=D['10_lags_report']||{};

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Análise Exploratória & Lags Biológicos'),
    el('p',null,'Correlações Spearman entre 12 variáveis climáticas (INMET) e notificações de dengue, sobre 1.224.164 registros. Fase 2 adicionou defasagens de 1-8 semanas alinhadas ao ciclo do Aedes aegypti.')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'Spearman (ρ): '),'Mede relação monotônica sem assumir linearidade. Valores |ρ| > 0.10 com p < 0.05 são significativos. O lag ótimo é a defasagem que maximiza a correlação — reflete o tempo entre condições climáticas favoráveis e o pico de notificações.'
  ));

  const sigFeats=eda.significant_features||[];
  e.appendChild(el('div',{cls:'co'},
    el('h3',null,(eda.n_features_significant||6)+' features significativas (|ρ| > 0.10)'),
    el('p',null,(sigFeats.length?sigFeats.join(', '):'humidity_mean_pct, temp_max_c, temp_min_c, temp_mean_c, rain_sum_mm, wind_speed_mean_ms')+'. Com lags biológicos: '+(lags.n_features_above_threshold||23)+' combinações feature×lag superam o limiar.')
  ));

  const lo=D.eda_lag_otimo||[];
  if(lo.length){
    mkChart(e,{title:'Lag ótimo por variável climática — ρ Spearman máximo',type:'bar',
      data:{labels:lo.map(r=>r.feature),datasets:[{label:'ρ Spearman',
        data:lo.map(r=>parseFloat(r.r||0)),
        backgroundColor:lo.map(r=>Math.abs(parseFloat(r.r||0))>0.1?c1():'rgba(42,120,214,.3)'),borderRadius:4,maxBarThickness:28}]},
      plugins:{legend:{display:false}},
      tooltipOpts:{callbacks:{label:c=>'ρ = '+c.raw.toFixed(4)+' (lag '+lo[c.dataIndex].best_lag+'sem)'}}
    });
    mkTbl(e,{title:'Lag ótimo por feature',search:false,
      hdr:['Feature','Lag (sem)','ρ','p-value','Significativo'],
      rows:lo.map(r=>[r.feature,r.best_lag,fmt(r.r),pv(r.p_value),Math.abs(parseFloat(r.r||0))>0.10?'✓':'—'])
    });
  }

  const lb=D.eda_lags_bio||[];
  if(lb.length){
    const feats=[...new Set(lb.map(r=>r.feature_base))];
    feats.slice(0,3).forEach(feat=>{
      const fd=lb.filter(r=>r.feature_base===feat&&r.type==='lag').sort((a,b)=>parseInt(a.lag)-parseInt(b.lag));
      if(!fd.length)return;
      mkChart(e,{title:feat+' — ρ por defasagem (semanas)',type:'bar',aspect:2.5,
        data:{labels:fd.map(r=>r.lag+'w'),datasets:[{label:'ρ',
          data:fd.map(r=>parseFloat(r.spearman_r||0)),
          backgroundColor:fd.map(r=>parseFloat(r.spearman_r||0)>0?c1():c2()),borderRadius:4,maxBarThickness:24}]},
        plugins:{legend:{display:false}}
      });
    });
  }

  const er=D.eda_regiao||[];
  if(er.length){mkTbl(e,{title:'Correlações por região',hdr:['Feature','Região','Lag','ρ','p-value','|ρ|'],
    rows:er.map(r=>[r.feature,r.regiao,r.lag,fmt(r.r),pv(r.p_value),fmt(r.abs_r)])})}

  const sp=D.eda_spearman||[];
  if(sp.length){mkTbl(e,{title:'Matriz Spearman completa',hdr:['Feature','Lag','ρ','p-value','n','Sig.'],
    rows:sp.map(r=>[r.feature,r.lag,fmt(r.r),pv(r.p_value),parseInt(r.n||0).toLocaleString(),r.sig||''])})}
},

regression(){
  const e=document.getElementById('s-regression');
  const rv=D['13_regression_v2_report']||{};
  const m=rv.modelos||{};
  const tt=rv.testes_t||{};

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Regressão XGBoost — 3 Modelos'),
    el('p',null,'Target: log1p(notificações t+4). A (SINAN-only, 129 feat), B (+ INMET bruto, 159), C (+ lags + anomalias, 303). Teste: 2024-2026, 515.764 linhas.')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'Feature engineering não ajuda a regressão: '),
    'O modelo B (INMET bruto) supera o C (enriquecido). Os 96 lags biológicos e 24 anomalias adicionam ruído com features SINAN presentes. Para prever o número exato de casos, o histórico epidemiológico domina.'
  ));

  e.appendChild(el('div',{cls:'co'},
    el('h3',null,'Modelo B vence: R² = '+fmt(m.B_inmet_bruto?.R2_log)+' (Δ = +'+fmt((m.B_inmet_bruto?.R2_log||0)-(m.A_sinan_only?.R2_log||0))+' vs SINAN-only)'),
    el('p',null,'Melhora estatisticamente significativa (p < 10⁻¹⁸), mas marginal em magnitude. Modelo C: Δ = +'+fmt((m.C_inmet_enriquecido?.R2_log||0)-(m.A_sinan_only?.R2_log||0))+' vs A, não significativo (p = 0.218).')
  ));

  const mk=['A_sinan_only','B_inmet_bruto','C_inmet_enriquecido'];
  const ml=['A: SINAN-only','B: INMET bruto','C: INMET enriquecido'];

  mkChart(e,{title:'R² log — Comparação dos 3 modelos',type:'bar',
    data:{labels:ml,datasets:[{label:'R² log',data:mk.map(k=>m[k]?.R2_log||0),
      backgroundColor:[c1(),'#eda100',c2()],borderRadius:4,maxBarThickness:40}]},
    yAxis:{min:Math.min(...mk.map(k=>m[k]?.R2_log||0))-0.003},
    plugins:{legend:{display:false}},
    tooltipOpts:{callbacks:{label:c=>'R² log: '+c.raw.toFixed(4)}}
  });

  mkTbl(e,{title:'Métricas detalhadas',search:false,
    hdr:['Modelo','Features','R² log','RMSE','MAE','R² orig'],
    rows:mk.map((k,i)=>[ml[i],String(m[k]?.n_features||'?'),fmt(m[k]?.R2_log),f2(m[k]?.RMSE),f2(m[k]?.MAE),fmt(m[k]?.R2_orig)])
  });

  if(Object.keys(tt).length){
    mkTbl(e,{title:'Testes t pareados',search:false,
      hdr:['Comparação','t-statistic','p-value','ΔR²','Significativo'],
      rows:Object.entries(tt).map(([n,t])=>[n,f2(t.t_stat),pv(t.p_value),(t.delta_R2_log>0?'+':'')+fmt(t.delta_R2_log),
        parseFloat(t.p_value||1)<0.05?'✓ Sim':'Não'])
    });
  }

  const uf=D.metrics_uf_v2||[];
  if(uf.length){mkTbl(e,{title:'Métricas por UF — 3 modelos',
    hdr:['UF','Região','Modelo','R² log','RMSE','n'],
    rows:uf.map(r=>[r.uf,r.regiao,r.modelo,fmt(r.R2_log),f2(r.RMSE),parseInt(r.n||0).toLocaleString()])})}
},

classification(){
  const e=document.getElementById('s-classification');
  const cv=D['15_classification_v2_report']||{};
  const m=cv.modelos||{};
  const labels=Object.keys(m);
  const classes=['baixo','médio','alto','surto'];

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Classificação de Risco — 4 Classes'),
    el('p',null,'XGBClassifier com 4 classes (baixo/médio/alto/surto) definidas por percentis municipais. Sample weights inversamente proporcionais à frequência para balanceamento.')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'Por que classificação: '),
    'Para gestores, saber se haverá surto é mais acionável que prever o número exato. Comprar inseticida e preparar leitos depende do nível de risco. Aqui, diferente da regressão, o INMET enriquecido (C) vence.'
  ));

  e.appendChild(el('div',{cls:'co im'},
    el('h3',null,'INMET enriquecido vence (F1 macro = '+fmt(m['C: INMET enriquecido']?.f1_macro||0)+')'),
    el('p',null,'Na classificação, modelo C com lags biológicos e anomalias é o melhor. O clima funciona como gatilho de categoria de risco, não como preditor quantitativo.')
  ));

  if(labels.length){
    const colors=[c1(),'#eda100',c2()];
    mkChart(e,{title:'F1-Score por classe de risco',type:'bar',
      data:{labels:classes,datasets:labels.map((l,i)=>({label:l,
        data:classes.map(c=>m[l]?.f1_per_class?.[c]||0),backgroundColor:colors[i]||c1(),borderRadius:4,maxBarThickness:24}))},
      tooltipOpts:{callbacks:{label:c=>c.dataset.label+': '+c.raw.toFixed(3)}}
    });

    mkTbl(e,{title:'Métricas de classificação',search:false,
      hdr:['Modelo','Acurácia','F1 macro','F1 weighted','AUC macro','F1 surto','Features'],
      rows:labels.map(l=>{const x=m[l];return[l,fmt(x?.accuracy),fmt(x?.f1_macro),fmt(x?.f1_weighted),fmt(x?.auc_macro),fmt(x?.f1_per_class?.surto),String(x?.n_features||'?')]})
    });
  }

  const cmA=m['A: SINAN-only']?.confusion_matrix;
  if(cmA){
    const w=el('div',{cls:'cw'},el('h3',null,'Matriz de confusão — Modelo A (referência)'));
    const t=el('table');
    t.appendChild(el('thead',null,el('tr',null,el('th',null,'Real \\ Prev'),...classes.map(c=>el('th',null,c)))));
    const tb=el('tbody');
    cmA.forEach((row,i)=>{
      const tot=row.reduce((a,b)=>a+b,0);
      tb.appendChild(el('tr',null,el('td',{sty:{fontFamily:'var(--ff-b)',fontWeight:'600'}},classes[i]),
        ...row.map((v,j)=>{
          const pct=tot>0?(v/tot*100).toFixed(0):'0';
          const td=el('td',null,v.toLocaleString()+' ('+pct+'%)');
          if(i===j){td.style.fontWeight='600';if(parseFloat(pct)>50)td.style.color='var(--ok)'}
          return td;
        })
      ));
    });
    t.appendChild(tb);w.appendChild(t);e.appendChild(w);
  }
},

shap(){
  const e=document.getElementById('s-shap');
  const sv=D['16_shap_v2_report']||{};
  const feats=D.shap_v2||[];
  const thresh=sv.thresholds||[];

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'SHAP — Explicabilidade & Limiares'),
    el('p',null,'TreeExplainer em 50K amostras do melhor modelo de regressão ('+(sv.modelo_analisado||'?')+', R² = '+fmt(D['13_regression_v2_report']?.modelos?.B_inmet_bruto?.R2_log||0)+').')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'SHAP: '),'Atribui a cada feature uma contribuição marginal para a previsão. O limiar SHAP é o valor onde a contribuição muda de proteção (negativa) para risco (positiva). Features SINAN dominam, mas as climáticas no top-30 revelam padrões biológicos.'
  ));

  const kg=el('div',{cls:'kpi-grid k4'});
  [[String(sv.climate_in_top_10||0),'Clima no Top-10','de '+sv.n_features+' features',''],
   [String(sv.climate_in_top_20||0),'Clima no Top-20','SHAP médio: '+fmt(sv.mean_shap_inmet||0),''],
   [String(sv.climate_in_top_30||0),'Clima no Top-30','SHAP SINAN: '+fmt(sv.mean_shap_sinan||0),''],
   [String(sv.best_climate_rank||'?'),'Rank melhor clima',sv.best_climate_feature||'?','']
  ].forEach(([v,l,s,c])=>kg.appendChild(el('div',{cls:'kpi'},el('div',{cls:'kpi-v'},v),el('div',{cls:'kpi-l'},l),el('div',{cls:'kpi-d '+c},s))));
  e.appendChild(kg);

  if(thresh.length){
    e.appendChild(el('h3',{sty:{margin:'16px 0 10px',fontSize:'.9rem'}},'Limiares Climáticos — Pontos de Inflexão'));
    e.appendChild(el('div',{cls:'note'},'Cada limiar indica onde o impacto SHAP cruza zero — abaixo, a variável protege; acima, aumenta o risco. Valores biologicamente interpretáveis para sistemas de alerta.'));
    const tg=el('div',{cls:'thr-g'});
    thresh.forEach(t=>{
      const arrow=t.direction==='positivo'?'↑ Acima = risco':'↓ Acima = proteção';
      tg.appendChild(el('div',{cls:'thr'},
        el('div',{cls:'nm'},t.feature.replace(/_/g,' ')),
        el('div',{cls:'tv'},String(t.threshold)),
        el('div',{cls:'dr'},arrow),
        el('div',{sty:{fontSize:'.7rem',color:'var(--tx3)',marginTop:'3px',fontFamily:'var(--ff-m)'}},'SHAP ↑'+(t.mean_shap_above>0?'+':'')+fmt(t.mean_shap_above)+' / ↓'+(t.mean_shap_below>0?'+':'')+fmt(t.mean_shap_below))
      ));
    });
    e.appendChild(tg);
  }

  const t30=feats.slice(0,30);
  if(t30.length){
    mkChart(e,{title:'Top-30 Features — Importância SHAP',type:'bar',aspect:1.2,
      data:{labels:t30.map(r=>r.feature),datasets:[{label:'Mean |SHAP|',
        data:t30.map(r=>parseFloat(r.mean_abs_shap||0)),
        backgroundColor:t30.map(r=>r.is_inmet==='True'?c2():c1()),borderRadius:4,maxBarThickness:16}]},
      opts:{indexAxis:'y'},
      yAxis:{ticks:{font:{family:"'IBM Plex Mono'",size:9}}},
      plugins:{legend:{display:false}},
      tooltipOpts:{callbacks:{label:c=>{const ft=t30[c.dataIndex];return (ft.is_inmet==='True'?'[INMET] ':'[SINAN] ')+c.raw.toFixed(4)}}}
    });
  }

  if(feats.length){
    mkTbl(e,{title:'Ranking completo SHAP',
      hdr:['Rank','Feature','SHAP','Fonte','Anomalia','Lag Bio','MM Bio'],
      rows:feats.map(r=>{
        const src=r.is_inmet==='True'?el('span',{cls:'bg im'},'INMET'):el('span',{cls:'bg si'},'SINAN');
        return[r.rank,r.feature,fmt(r.mean_abs_shap),src,
          r.is_anomaly==='True'?'Sim':'',r.is_lag_bio==='True'?'Sim':'',r.is_mm_bio==='True'?'Sim':'']
      })
    });
  }
},

walkforward(){
  const e=document.getElementById('s-walkforward');
  const wv=D['17_walkforward_v2_report']||{};
  const fs=wv.sinan_only?.folds||[];
  const fi=wv.inmet_enriquecido?.folds||[];

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Validação Walk-Forward — 5 Folds'),
    el('p',null,'Validação temporal com janela expansiva: treina com anos anteriores, testa no próximo. Garante que resultados não são artefato de um único split.')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',null,'Funcionamento: '),
    'Fold 1: treina em 2019, testa em 2021. Fold 2: treina 2019+2021, testa 2023. E assim por diante. Sempre prevê o futuro usando apenas dados passados.'
  ));

  const delta=wv.delta_mean||0;
  e.appendChild(el('div',{cls:'co '+(delta>0?'ok':'bd')},
    el('h3',null,'INMET ganha '+(wv.folds_inmet_better||0)+'/5 folds, mas perde na média (Δ = '+fmt(delta)+')'),
    el('p',null,'R² médio SINAN-only: '+fmt(wv.sinan_only?.mean_R2_log||0)+' ± '+fmt(wv.sinan_only?.std_R2_log||0)+
      ' vs INMET: '+fmt(wv.inmet_enriquecido?.mean_R2_log||0)+' ± '+fmt(wv.inmet_enriquecido?.std_R2_log||0)+
      '. Vence em 2021, 2023, 2024, mas perde em 2025 (-0.0285) e 2026 (-0.0350) por anomalias climáticas.')
  ));

  if(fs.length){
    const years=fs.map(x=>String(x.test_year));
    const dark=isDark();
    const surfC=dark?'#161B25':'#FFFFFF';

    mkChart(e,{title:'R² log por fold — SINAN-only vs INMET enriquecido',type:'line',
      data:{labels:years,datasets:[
        {label:'SINAN-only',data:fs.map(x=>x.R2_log),borderColor:c1(),backgroundColor:'rgba(42,120,214,.08)',
          fill:true,borderWidth:2,pointRadius:5,pointHoverRadius:8,pointBackgroundColor:c1(),pointBorderColor:surfC,pointBorderWidth:2},
        {label:'INMET enriquecido',data:fi.map(x=>x.R2_log),borderColor:c2(),backgroundColor:'rgba(235,104,52,.08)',
          fill:true,borderWidth:2,pointRadius:5,pointHoverRadius:8,pointBackgroundColor:c2(),pointBorderColor:surfC,pointBorderWidth:2}
      ]},
      tooltipOpts:{callbacks:{label:c=>c.dataset.label+': '+c.raw.toFixed(4)}}
    });

    const deltas=fs.map((s,i)=>(fi[i]?.R2_log||0)-s.R2_log);
    mkChart(e,{title:'ΔR² por fold (INMET − SINAN-only)',type:'bar',aspect:2.5,
      data:{labels:years,datasets:[{label:'ΔR²',data:deltas,
        backgroundColor:deltas.map(d=>d>0?'#0ca30c':'#d03b3b'),borderRadius:4,maxBarThickness:28}]},
      plugins:{legend:{display:false}},
      tooltipOpts:{callbacks:{label:c=>'Δ: '+(c.raw>0?'+':'')+c.raw.toFixed(4)}}
    });

    mkTbl(e,{title:'Detalhes por fold',search:false,
      hdr:['Ano','Treino','R² SINAN','R² INMET','ΔR²','RMSE SINAN','RMSE INMET','Treino n','Teste n'],
      rows:fs.map((s,i)=>{
        const x=fi[i]||{};const d=(x.R2_log||0)-s.R2_log;
        return[String(s.test_year),String(s.train_years||''),fmt(s.R2_log),fmt(x.R2_log),(d>0?'+':'')+fmt(d),
          f2(s.RMSE),f2(x.RMSE),(s.train_rows||0).toLocaleString(),(s.test_rows||0).toLocaleString()]
      })
    });
  }
},

methodology(){
  const e=document.getElementById('s-methodology');

  e.appendChild(el('div',{cls:'sec-head'},
    el('h2',null,'Metodologia'),
    el('p',null,'Pipeline completo de coleta, integração, filtragem, feature engineering e modelagem. Dados reais do SINAN (Ministério da Saúde) e INMET (Instituto Nacional de Meteorologia).')
  ));

  e.appendChild(el('div',{cls:'note'},
    el('strong',{sty:{color:'var(--c1)'}},'SINAN'),' — Notificações de dengue por município e semana epidemiológica. ',
    el('strong',{sty:{color:'var(--c2)'}},'INMET'),' — Dados horários de 568 estações meteorológicas automáticas, agregados para nível municipal/semanal via mapeamento geodésico (≤ 50km).'
  ));

  const stages=[
    ['Coleta','SINAN: DataSUS (33.4M notificações, 5.565 municípios). INMET: API (567 estações, 24 anos).','Pipeline Medallion: Bronze → Silver → Gold'],
    ['Integração','JOIN por (ibge_municipio, ano, semana) com mapeamento estação↔município por haversine ≤ 50km + IDW.','7.665.428 linhas, 172 features'],
    ['Filtragem','4.406 municípios com estação INMET ≤ 50km. 6 anos com cobertura > 80% (excluídos 2020/2022).','1.224.164 linhas, 315 colunas'],
    ['Feature Engineering','Fase 1: 129 SINAN + 30 INMET bruto. Fase 2: + 96 lags (12×8) + 24 médias móveis + 24 anomalias.','303 features totais'],
    ['Splits Temporais','Train: 2019+2021 (457.600). Val: 2023 (233.200). Test: 2024-2026 (515.764). Sem data leakage.','Cobertura INMET > 85% por split'],
    ['Target','notificacoes_t4 = shift(-4) por município. Classificação: 4 classes via percentis p50/p75/p90.','Horizonte: t+4 semanas'],
  ];
  const sg=el('div',{cls:'st-g'});
  stages.forEach(([t,d,dt])=>sg.appendChild(el('div',{cls:'st'},
    el('h3',null,t),el('p',null,d),el('p',{cls:'meta'},dt))));
  e.appendChild(sg);

  mkTbl(e,{title:'Hiperparâmetros XGBoost',search:false,
    hdr:['Parâmetro','Regressão','Classificação'],
    rows:[['n_estimators','1000','800'],['max_depth','8','8'],['learning_rate','0.05','0.05'],
      ['subsample','0.8','0.8'],['colsample_bytree','0.8','0.8'],['reg_alpha','0.1','0.1'],
      ['reg_lambda','1.0','1.0'],['min_child_weight','5','5'],['tree_method','hist','hist'],
      ['early_stopping_rounds','50','50'],['objective','reg:squarederror','multi:softprob (4 classes)'],
      ['target','log1p(notificacoes_t4)','risco_surto_t4'],['random_state','42','42']]
  });

  mkTbl(e,{title:'Pipeline de scripts — Fase 2',search:false,
    hdr:['Script','US','Descrição','Status'],
    rows:[
      ['10_feature_eng_lags.py','110','96 lags (12×8) + 24 médias móveis biológicas','✓'],
      ['11_feature_eng_anomalias.py','111','24 anomalias (desvio da média histórica)','✓'],
      ['12_prepare_splits_v2.py','112','Splits com 303 features enriquecidas','✓'],
      ['13_train_regression_v2.py','113','3 modelos regressão: A, B, C','✓'],
      ['14_train_blindness.py','114','4 cenários × 2 variantes de Data Blindness','✓'],
      ['15_train_classification_v2.py','115','3 XGBClassifier, 4 classes + sample weights','✓'],
      ['16_explain_shap_v2.py','116','TreeExplainer 50K + limiares','✓'],
      ['17_validate_walkforward_v2.py','117','5 folds walk-forward temporal','✓'],
      ['18_compile_final.py','118','Consolidação + tabelas LaTeX','✓'],
    ]
  });
}

};

P.overview();R.overview=1;
</script>
"""

html = html_template.replace('__DATA__', data_json)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fout:
    fout.write(html)

print(f"Dashboard v3: {OUT}")
print(f"Size: {len(html):,} chars ({len(html)//1024} KB)")
