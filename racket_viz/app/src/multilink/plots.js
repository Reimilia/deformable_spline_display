export const COLORS = { a: "#326af0", b: "#be6999", target: "#d39b3a", initial: "#bdc7d5", kinetic: "#4bafaa", potential: "#d8a044", grid: "#eff2f6", text: "#8895a6" };

export function bounds(points, equal = false, width = 1, height = 1) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x,y] of points) { if (!Number.isFinite(x+y)) continue; xmin=Math.min(xmin,x); xmax=Math.max(xmax,x); ymin=Math.min(ymin,y); ymax=Math.max(ymax,y); }
  if (!Number.isFinite(xmin)) return [-1,1,-1,1];
  const dx = Math.max(xmax-xmin, 0.02), dy = Math.max(ymax-ymin, 0.02);
  let spanX = dx * 1.25, spanY = dy * 1.25;
  if (equal) { const scale = Math.max(spanX/width, spanY/height); spanX=scale*width; spanY=scale*height; }
  return [(xmin+xmax-spanX)/2, (xmin+xmax+spanX)/2, (ymin+ymax-spanY)/2, (ymin+ymax+spanY)/2];
}

export function format(v) { if (!Number.isFinite(v)) return "—"; const a=Math.abs(v); return a!==0 && (a < .001 || a>=10000) ? v.toExponential(2) : v.toFixed(a<.1?4:a<10?3:1); }

export function frameAt(times, time) {
  let lo=0, hi=times.length-1;
  while (lo<hi) { const mid=Math.ceil((lo+hi)/2); if(times[mid]<=time) lo=mid; else hi=mid-1; }
  return lo;
}

export function validateRollout(data) {
  if (data.schema_version!==1 || !Array.isArray(data.runs) || !data.runs.length) throw new Error("Invalid rollout response");
  for(const r of data.runs) {
    const n=r.times?.length;
    if(!n || n<2 || r.world.length!==n || r.body.length!==n || r.phase_q.length!==n || r.phase_p.length!==n) throw new Error("Inconsistent rollout frames");
    if(r.times.some((t,i)=>!Number.isFinite(t)||(i&&t<=r.times[i-1]))) throw new Error("Invalid simulation times");
    if(Object.values(r.metrics).some(v=>v.length!==n-1 || v.some(x=>!Number.isFinite(x)))) throw new Error("Inconsistent metric frames");
  }
  return data;
}

export function prepare(canvas) {
  const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const w=Math.max(rect.width,1),h=Math.max(rect.height,1);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)) { canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr); }
  const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  ctx.font="9px system-ui"; return {ctx,w,h};
}

export function axes(canvas, domain, xLabel, yLabel) {
  const {ctx,w,h}=prepare(canvas),pad={l:49,r:15,t:16,b:34};
  const pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
  const [xmin,xmax,ymin,ymax]=domain;
  const map=([x,y])=>[pad.l+(x-xmin)/(xmax-xmin)*pw,pad.t+(ymax-y)/(ymax-ymin)*ph];
  ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;
  for(let i=0;i<=4;i++) {
    const x=pad.l+pw*i/4,y=pad.t+ph*i/4;
    ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+ph);ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+pw,y);ctx.stroke();
    ctx.fillStyle=COLORS.text;ctx.textAlign="center";ctx.fillText(format(xmin+(xmax-xmin)*i/4),x,h-18);
    ctx.textAlign="right";ctx.fillText(format(ymax-(ymax-ymin)*i/4),pad.l-7,y+3);
  }
  ctx.textAlign="center";ctx.fillText(xLabel,pad.l+pw/2,h-3);
  ctx.save();ctx.translate(11,pad.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
  return {ctx,map,w,h,pad,pw,ph};
}

export function line(plot, points, color, width=1.8, dash=[], opacity=1) {
  if(!points?.length) return;
  const {ctx,map}=plot;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.globalAlpha=opacity;ctx.beginPath();
  points.forEach((p,i)=>{const [x,y]=map(p);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.restore();
}

export function dot(plot, point, color, radius=3.5) {
  const [x,y]=plot.map(point); plot.ctx.beginPath();plot.ctx.arc(x,y,radius,0,2*Math.PI);plot.ctx.fillStyle=color;plot.ctx.fill();
}

export function cursor(plot, time) {
  const [x]=plot.map([time,0]);plot.ctx.save();plot.ctx.strokeStyle="#a4b2c4";plot.ctx.setLineDash([3,4]);plot.ctx.beginPath();plot.ctx.moveTo(x,plot.pad.t);plot.ctx.lineTo(x,plot.pad.t+plot.ph);plot.ctx.stroke();plot.ctx.restore();
}
