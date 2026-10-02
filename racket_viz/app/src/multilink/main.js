import "./style.css";
import "./service.css";
import { Playback } from "../core/Playback.js";
import { COLORS, bounds, format, frameAt, validateRollout, axes, line, dot, cursor, prepare } from "./plots.js";
import { assetUrl, normalizeApiBase, apiUrl, fetchJson, previewFor, comparisonOptions, combinePreviews } from "./source.js";

const $=id=>document.getElementById(id);
let catalog=[],scenario=null,playback=null,busy=false,mode="live",manifest=null,apiBase="";
const base=import.meta.env.BASE_URL;
const storageKey=`deformation-lab:${location.pathname}`;
const previewCache=new Map();
const physicsIds=["steps","horizon","damping","stiffness","target-dx","target-dy","target-angle","initial-angle","noise","seed","tolerance","fim"];
const groupNames={test:"Feasible test",hard_test:"Hard OOD test",hard_adapt:"Hard adaptation",train_stage1:"Stage 1 train",val_stage1:"Stage 1 validation",train_stage2:"Stage 2 train",val_stage2:"Stage 2 validation"};
const colors=[COLORS.a,COLORS.b];
const selected=()=>catalog.find(e=>e.id===$("checkpoint").value);
const familyEntries=()=>catalog.filter(e=>e.family===$("family").value);
const status=(text,kind="")=>{$("status").textContent=text;$("status").className=`status ${kind}`;};

function fill(select, options, value) {
  select.replaceChildren(...options.map(([v,t])=>new Option(t,v)));
  if(options.some(([v])=>v===value)) select.value=value;
}

function populateModels() {
  const es=familyEntries();
  fill($("checkpoint"),es.map(e=>[e.id,e.label]));
  const initial=es.find(e=>e.mode_label==="FIM" && e.stage==="Final" && (e.family==="multilink"||e.conditioning==="observation"))||es.find(e=>e.stage==="Final")||es[0];
  $("checkpoint").value=initial.id;checkpointChanged();
}

function populateTasks() {
  const e=selected(), g=$("group").value;
  fill($("task-index"),Array.from({length:e.tasks[g]},(_,i)=>[String(i),`Task ${i+1}`]));
}

function populateCompare() {
  const e=selected(),value=$("compare").value;
  fill($("compare"),[["","Single checkpoint"],...comparisonOptions(familyEntries(),e,mode).map(x=>[x.id,x.label])],value);
}

function availability() {
  document.querySelectorAll("aside input,aside select,aside button").forEach(el=>el.disabled=busy);
  if(mode==="preview")for(const id of physicsIds)$(id).disabled=true;
  $("source-mode").textContent=mode==="preview"?"Saved previews":"Live dynamics";
  $("run").innerHTML=mode==="preview"?"Load preview <span>→</span>":"Run dynamics <span>→</span>";
  $("source-note").textContent=mode==="preview"?`Saved checkpoint rollouts · ${manifest.horizon}× horizon. Connect a dynamics service to change physics. Compare stages within the same variant.`:"Command interval: 1 normalized time unit. The remaining horizon shows settling.";
}

function previewControls() {
  if(mode!=="preview")return;
  for(const id of ["target-dx","target-dy","target-angle","initial-angle","noise"])$(id).value=0;
  $("horizon").value=manifest.horizon;$("damping").value=1;$("stiffness").value=1;$("seed").value=2026;
  refreshSliderLabels();
}

function checkpointChanged() {
  const e=selected(),groups=Object.keys(e.tasks);
  fill($("group"),groups.map(g=>[g,groupNames[g]||g]),e.stage==="Hard transfer"?"hard_test":"test");
  populateTasks();populateCompare();
  $("steps").value=e.defaults.steps;$("tolerance").value=e.defaults.tolerance;$("fim").checked=e.defaults.fim;
  $("tolerance-label").hidden=e.mode_label!=="FIM";
  $("fim-row").hidden=e.family!=="multilink";
  $("noise-row").hidden=e.family!=="spline";
  $("checkpoint-info").textContent=`${e.mode_label} · ${e.stage} · ${e.conditioning} conditioning. ${e.family==="multilink"?"14 links, 13 joint coordinates.":"18 control points, learned knot spans."}`;
  previewControls();availability();
  markChanged();
}

function resetControls() {
  for(const id of ["target-dx","target-dy","target-angle","initial-angle","noise"]) $(id).value=0;
  $("horizon").value=2;$("damping").value=1;$("stiffness").value=1;$("seed").value=2026;
  $("compare").value="";checkpointChanged();refreshSliderLabels();
}

function markChanged() { if(scenario&&!busy)status(mode==="preview"?"Selection changed · Load preview to update the trajectory":"Configuration changed · Run to update the trajectory"); }
function refreshSliderLabels() { for(const id of ["horizon","damping","stiffness"]) $(`${id}-val`).textContent=`${Number($(id).value)}×`; }

function readConfig() {
  const map={steps:"steps",horizon:"horizon",damping:"damping",stiffness:"stiffness",target_dx:"target-dx",target_dy:"target-dy",target_angle:"target-angle",initial_angle:"initial-angle",noise:"noise",seed:"seed",tolerance:"tolerance"};
  const data={checkpoint:$("checkpoint").value,compare:$("compare").value,group:$("group").value,task_index:Number($("task-index").value),fim:$("fim").checked};
  for(const [key,id] of Object.entries(map)) { const el=$(id);if(!el.checkValidity())throw new Error(`${el.parentElement.textContent.trim().split("\n")[0]}: enter a value in the allowed range`);data[key]=Number(el.value); }
  return data;
}

function legend(id,items) {
  $(id).replaceChildren(...items.map(([label,color])=>{const span=document.createElement("span"),swatch=document.createElement("i");swatch.style.background=color;span.append(swatch,document.createTextNode(label));return span;}));
}
function runName(run,i) {return `${i===0?"A":"B"}: ${run.mode_label} · ${run.label.split(" · ").at(-1)}`;}

async function savedRun(config) {
  async function get(id) {
    const item=previewFor(manifest,{...config,checkpoint:id}),url=assetUrl(`previews/${item.file}`,base,location.href);
    if(!previewCache.has(url)) {
      const result=validateRollout(await fetchJson(url));
      if(previewCache.size>=8)previewCache.delete(previewCache.keys().next().value);
      previewCache.set(url,result);
    }
    return previewCache.get(url);
  }
  const a=await get(config.checkpoint);
  return config.compare?combinePreviews(a,await get(config.compare)):a;
}

async function run() {
  if(busy)return;
  let config;
  try{config=readConfig();}catch(e){status(e.message,"error");return;}
  busy=true;playback?.pause();
  availability();
  status(mode==="preview"?"Loading saved checkpoint rollout…":"Integrating selected checkpoint…","busy");
  const start=performance.now();
  try {
    const result=mode==="preview"?await savedRun(config):await fetchJson(apiUrl("rollout",apiBase,base,location.href),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(config)});
    scenario=validateRollout(result);
    const longest=scenario.runs.reduce((a,b)=>a.times.at(-1)>b.times.at(-1)?a:b);
    playback=new Playback(longest.times);playback.play();
    $("scrub").max=longest.times.at(-1);$("play").disabled=false;$("export").disabled=false;
    fill($("coordinate"),scenario.runs[0].coordinates.map((name,i)=>[String(i),name]));
    for(const id of ["energy-view","error-view"]) {
      for(const option of $(id).options)option.disabled=option.value!=="energy"&&!scenario.runs.every(r=>option.value in r.metrics);
      if($(id).selectedOptions[0].disabled)$(id).value=id==="energy-view"?"energy":"shape_error";
    }
    $("run-title").textContent=`${scenario.family==="multilink"?"Articulated loop":"Spline curve"} · ${groupNames[scenario.group]||scenario.group} ${scenario.task_index+1}`;
    $("run-notes").textContent=scenario.runs[0].notes+(scenario.runs.length>1?" Overlay uses the same saved task; B keeps its trained closure tolerance and FIM setting.":"")+(mode==="preview"?" Saved rollout; changing the view or playback does not recompute dynamics.":"");
    const diverged=scenario.runs.some(r=>r.diverged);
    status(diverged?"Dynamics diverged · showing the finite trajectory before termination":`${scenario.runs.length===2?"Two checkpoints":"Checkpoint"} ${mode==="preview"?"preview loaded":"integrated"} · ${((performance.now()-start)/1000).toFixed(2)} s`,diverged?"error":"");
    draw();
  } catch(e) {status(`Unable to ${mode==="preview"?"load preview":"run"}: ${e.message}`,"error");}
  finally {busy=false;availability();}
}

function drawShape(time) {
  const frame=$("frame").value,runs=scenario.runs,canvas=$("shape-canvas");
  const {w,h}=prepare(canvas);
  const all=runs.flatMap(r=>[...r[frame].flat(),...r[`target_${frame}`]]);
  if($("show-reference").checked) all.push(...runs.flatMap(r=>r[`reference_${frame}`].flat()));
  if(frame==="world"&&$("show-observations").checked)all.push(...runs.flatMap(r=>r.observations_world||[]));
  const plot=axes(canvas,bounds(all,true,w-64,h-50),"x","y");
  runs.forEach((r,i)=>{
    const idx=frameAt(r.times,time),color=colors[i];
    line(plot,r[frame][0],COLORS.initial,1,[3,4]);
    line(plot,r[`target_${frame}`],i===0?COLORS.target:COLORS.b,1.6,[5,4],i===0?1:.5);
    if(frame==="world"&&$("show-trail").checked)line(plot,r.poses.slice(0,idx+1).map(p=>p.slice(0,2)),color,1,[2,3],.5);
    if($("show-reference").checked)line(plot,r[`reference_${frame}`][Math.min(Math.max(idx-1,0),r.reference_body.length-1)],color,1,[4,4],.4);
    if(frame==="world"&&$("show-observations").checked)for(const p of (r.observations_world||[]).filter((_,j)=>j%4===0))dot(plot,p,COLORS.target,1);
    line(plot,r[frame][idx],color,2.6);
    if(scenario.family==="multilink")r[frame][idx].forEach(p=>dot(plot,p,color,2.5));
    // Actual endpoints reveal seam opening; the renderer never adds a closing edge.
    const pts=r[frame][idx];dot(plot,pts[0],color,4);dot(plot,pts.at(-1),color,3);
  });
  legend("shape-legend",[...runs.map((r,i)=>[runName(r,i),colors[i]]),["Target",COLORS.target],["Initial",COLORS.initial]]);
}

function drawPhase(time) {
  const c=Number($("coordinate").value),runs=scenario.runs;
  const pts=runs.map(r=>r.phase_q.map((q,j)=>[q[c],r.phase_p[j][c]]));
  const plot=axes($("phase-canvas"),bounds(pts.flat()),"q","p");
  runs.forEach((r,i)=>{const idx=frameAt(r.times,time);line(plot,pts[i],colors[i],1,[],.15);line(plot,pts[i].slice(0,idx+1),colors[i],1.8);dot(plot,pts[i][idx],colors[i],4);dot(plot,pts[i][0],COLORS.initial,2.5);});
  $("phase-description").textContent=`${runs[0].coordinates[c]} · configuration vs momentum`;
  legend("phase-legend",runs.map((r,i)=>[runName(r,i),colors[i]]));
}

function drawSeries(canvasId,legendId,key,time) {
  let series=[];
  scenario.runs.forEach((r,i)=>{
    const keys=key==="energy"?["hamiltonian","kinetic","potential"]:[key];
    for(const k of keys) {
      if(!(k in r.metrics))continue;
      const color=key==="energy"&&i===0?{hamiltonian:colors[i],kinetic:COLORS.kinetic,potential:COLORS.potential}[k]:colors[i];
      series.push({points:r.metric_times.map((t,j)=>[t,r.metrics[k][j]]),color,label: key==="energy"?`${i===0?"A":"B"} ${k==="hamiltonian"?"H":k==="kinetic"?"K":"V"}`:runName(r,i),dash:i===1?[4,3]:[],run:r});
    }
  });
  const domain=bounds(series.flatMap(s=>s.points));domain[0]=0;domain[1]=Math.max(...scenario.runs.map(r=>r.times.at(-1)));
  const plot=axes($(canvasId),domain,"time",key==="energy"?"energy":key.replaceAll("_"," "));
  // Mark the end of the command interval independently of the playback cursor.
  const [commandX]=plot.map([1,0]);plot.ctx.save();plot.ctx.strokeStyle="#dbe2eb";plot.ctx.setLineDash([2,4]);plot.ctx.beginPath();plot.ctx.moveTo(commandX,plot.pad.t);plot.ctx.lineTo(commandX,plot.pad.t+plot.ph);plot.ctx.stroke();plot.ctx.restore();
  for(const s of series){line(plot,s.points,s.color,1,s.dash,.18);const idx=frameAt(s.run.times,time);line(plot,s.points.slice(0,idx),s.color,1.6,s.dash);if(idx>0)dot(plot,s.points[Math.min(idx-1,s.points.length-1)],s.color,2.5);}
  cursor(plot,time);legend(legendId,series.map(s=>[s.label,s.color]));
}

function draw() {
  if(!scenario||!playback)return;
  const t=playback.currentTime;
  drawShape(t);drawPhase(t);drawSeries("energy-canvas","energy-legend",$("energy-view").value,t);drawSeries("error-canvas","error-legend",$("error-view").value,t);
  const r=scenario.runs[0],index=Math.max(0,frameAt(r.times,t)-1);
  for(const [id,key] of [["shape-value","shape_error"],["gap-value","closure_gap"],["energy-value","hamiltonian"],["speed-value","speed"]])$(id).textContent=t<r.metric_times[0]?"—":format(r.metrics[key][Math.min(index,r.metrics[key].length-1)]);
  $("scrub").value=t;$("time").textContent=`${t.toFixed(2)} / ${playback.times.at(-1).toFixed(2)}`;$("play").textContent=playback.playing?"Ⅱ Pause":t>=playback.times.at(-1)?"↻ Replay":"▶ Play";
}

$("family").addEventListener("change",populateModels);$("checkpoint").addEventListener("change",checkpointChanged);
$("group").addEventListener("change",()=>{populateTasks();markChanged();});$("run").addEventListener("click",run);$("reset").addEventListener("click",resetControls);
document.querySelectorAll("aside input,aside select").forEach(el=>el.addEventListener("input",()=>{refreshSliderLabels();markChanged();}));
for(const id of ["coordinate","frame","energy-view","error-view","show-reference","show-observations","show-trail"])$(id).addEventListener("change",draw);
$("play").addEventListener("click",()=>{if(!playback)return;if(playback.playing)playback.pause();else{if(playback.currentTime>=playback.times.at(-1))playback.seek(0);playback.play();}draw();});
$("scrub").addEventListener("input",()=>{if(!playback)return;playback.pause();playback.seek(Number($("scrub").value));draw();});
$("export").addEventListener("click",()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(scenario)],{type:"application/json"}));
  const a=document.createElement("a");a.href=url;a.download=`deformation-${scenario.family}-${scenario.group}-${scenario.task_index+1}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener("resize",draw);
let last=performance.now();
function animate(now) { const dt=Math.min((now-last)/1000,.1);last=now;if(playback?.playing){const active=playback.tick(dt*Number($("speed").value));if(!active&&$("loop").checked&&!busy){playback.seek(0);playback.play();}draw();}requestAnimationFrame(animate); }
requestAnimationFrame(animate);

async function init(){
  let saved={},candidate=import.meta.env.VITE_API_URL||"";
  try{saved=JSON.parse(localStorage.getItem(storageKey)||"{}");candidate=saved.api||candidate;}catch{}
  $("api-base").value=candidate;
  if(saved.mode==="preview"||(import.meta.env.VITE_STATIC_DEMO==="true"&&!candidate))return startPreview();
  try{await startLive(candidate);}catch{await startPreview();}
}

function remember(api,selectedMode){try{localStorage.setItem(storageKey,JSON.stringify({api,mode:selectedMode}));}catch{}}
function setCatalog(data){catalog=data.checkpoints;$("model-count").textContent=`${catalog.length} checkpoints`;populateModels();refreshSliderLabels();availability();}

async function startLive(input){
  const candidate=normalizeApiBase(input,location.href);
  status("Connecting to dynamics service…","busy");
  const data=await fetchJson(apiUrl("checkpoints",candidate,base,location.href),{signal:AbortSignal.timeout(8000)});
  if(!Array.isArray(data.checkpoints)||!data.checkpoints.length)throw new Error("The service returned no checkpoints");
  apiBase=candidate;mode="live";setCatalog(data);remember(apiBase,mode);await run();
}

async function startPreview(){
  try{
    manifest=await fetchJson(assetUrl("previews/manifest.json",base,location.href));
    if(manifest.schema_version!==1||!manifest.catalog?.checkpoints?.length)throw new Error("Saved preview catalog is invalid");
    mode="preview";apiBase="";setCatalog(manifest.catalog);remember("",mode);await run();
  }catch(e){status(`${e.message}. Generate saved previews or connect a dynamics service.`,"error");$("run").disabled=true;}
}

$("connect-api").addEventListener("click",async()=>{try{await startLive($("api-base").value);}catch(e){status(`Connection failed: ${e.message}`,"error");}});
$("use-previews").addEventListener("click",startPreview);
init();
