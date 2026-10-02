"""Float64 checkpoint inference and dynamics for CPython and Pyodide/WASM.

This evaluation port follows python/vendor/multilink_ph/{info_ph,spline_curve_ph}.
NumPy supplies WASM linear algebra; Autograd differentiates the finite-iteration
Sinkhorn and spline seam potentials. No Torch, API, or trajectory cache is used.
"""
import copy
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad
from autograd.tracer import getval


def logsumexp(x, axis):
    peak=anp.max(x,axis=axis,keepdims=True)
    return anp.squeeze(peak+anp.log(anp.sum(anp.exp(x-peak),axis=axis,keepdims=True)),axis=axis)


def wrap(x):
    return anp.arctan2(anp.sin(x), anp.cos(x))


def rotation(theta):
    c, s = anp.cos(theta), anp.sin(theta)
    return anp.array([[c, -s], [s, c]])


def transform(points, pose):
    return anp.dot(points, rotation(pose[2]).T) + pose[:2]


def compose(a, b):
    return np.r_[a[:2] + rotation(a[2]) @ b[:2], wrap(a[2] + b[2])]


def ab(theta):
    A = np.sinc(theta / math.pi)
    B = theta/2 - theta**3/24 + theta**5/720 if abs(theta) < 1e-6 else (1-np.cos(theta))/theta
    return A, B


def relative_log(a, b):
    xy = rotation(a[2]).T @ (b[:2] - a[:2])
    th = wrap(b[2] - a[2]); A, B = ab(th)
    return np.r_[(A*xy[0]+B*xy[1])/max(A*A+B*B, 1e-12),
                 (-B*xy[0]+A*xy[1])/max(A*A+B*B, 1e-12), th]


def mlp(state, prefix, x, layers, final_tanh=False):
    for i in range(layers):
        x = x @ state[f"{prefix}.{2*i}.weight"].T + state[f"{prefix}.{2*i}.bias"]
        if i < layers-1 or final_tanh:
            x = np.tanh(x)
    return x


def number(data, name, default, lo, hi, integer=False):
    try:
        x = float(data.get(name, default))
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(x) or not lo <= x <= hi or (integer and x != int(x)):
        raise ValueError(f"{name} must be {'an integer ' if integer else ''}between {lo} and {hi}")
    return int(x) if integer else x


def options(data, defaults):
    limits = {"steps":(defaults["steps"],12,240,True), "horizon":(2,1,6,False),
              "damping":(1,.1,5,False), "stiffness":(1,.2,3,False),
              "target_dx":(0,-2,2,False), "target_dy":(0,-2,2,False),
              "target_angle":(0,-180,180,False), "initial_angle":(0,-180,180,False),
              "noise":(0,0,.1,False), "seed":(2026,0,2**31-1,True),
              "tolerance":(defaults["tolerance"],.001,.3,False)}
    out = {k:number(data,k,*v) for k,v in limits.items()}
    out["fim"] = bool(data.get("fim", defaults["fim"]))
    return out


class Chain:
    def __init__(self, params):
        self.p = params
        self.m = params["n_links"]; self.j = self.m-1; self.n = self.j+3
        self.length = params["total_length"]/self.m
        self.mass = params["total_mass"]/self.m
        self.inertia = self.mass*self.length**2/12
        self.h = np.tril(np.ones((self.m, self.j)), -1)

    def kin(self, q):
        beta = np.r_[0, np.cumsum(q)]
        t = np.column_stack([np.cos(beta),np.sin(beta)])
        normal = np.column_stack([-t[:,1],t[:,0]])
        edge = t*self.length
        points = np.vstack([np.zeros(2),np.cumsum(edge,axis=0)])
        centers = points[:-1]+edge/2
        cumulative=np.cumsum(self.length*normal,axis=0)
        before=np.vstack([np.zeros(2),cumulative[:-1]])
        J=((before[:,None,:]-cumulative[None,:-1,:]+self.length*normal[:,None,:]/2)*self.h[:,:,None]).transpose(0,2,1)
        closureJ=(cumulative[-1]-cumulative[:-1]).T
        return points,centers,t,normal,J,closureJ

    def assemble(self, q):
        points,centers,t,normal,J,closureJ = self.kin(q)
        H=np.zeros((self.m,2,self.n)); H[:,:,:2]=np.eye(2)
        H[:,:,2]=np.column_stack([-centers[:,1],centers[:,0]]); H[:,:,3:]=J
        omega=np.column_stack([np.zeros((self.m,2)),np.ones(self.m),self.h])
        M=self.mass*np.einsum("mai,maj->ij",H,H)+self.inertia*(omega.T@omega)+np.eye(self.n)*self.p["mass_regularization"]
        tH=np.einsum("ma,mai->mi",t,H); nH=np.einsum("ma,mai->mi",normal,H)
        D=self.p["drag_scale"]/self.m*(self.p["c_parallel"]*(tH.T@tH)+self.p["c_perp"]*(nH.T@nH))
        D[:3,:3] += np.eye(3)*self.p["rigid_drag_regularization"]
        D[3:,3:] += np.eye(self.j)*self.p["joint_damping"]
        fim = np.einsum("mai,maj->ij",J,J)/self.m+.05*(self.h.T@self.h)/self.m+1e-4*np.eye(self.j)
        fim *= self.j/max(np.trace(fim),1e-4)
        return M,D,fim,points[-1],closureJ

    def project(self, q, iterations):
        for _ in range(iterations):
            points,_,_,_,_,J = self.kin(q)
            q = q-J.T@np.linalg.solve(J@J.T+1e-8*np.eye(2), points[-1])
        return q

    def seam(self, q, closure):
        points,_,_,_,_,J = self.kin(q)
        if closure["mode"] != "measurement":
            return 0.,np.zeros(self.j),np.zeros((self.j,self.j))
        gap = points[-1]
        sigma = max(closure["tolerance_rel"],1e-8)*self.p["total_length"]
        k = closure["energy_scale"]/(sigma*sigma+closure["eps"])
        d = closure["damping_scale"]*np.sqrt(max(k,0))
        return .5*k*(gap@gap), -J.T@(k*gap), d*(J.T@J)


def multilink(state, meta, task, entry, opt, compared, progress):
    chain = Chain(meta["chain_params"]); K = np.array(meta["Ksp"])
    closure = {"energy_scale":1e-3,"damping_scale":.12,"eps":1e-10,**meta["closure_model"]}
    if not compared:
        closure["tolerance_rel"] = opt["tolerance"]
    fim_on = meta["use_fim"] if compared else opt["fim"]
    stiffness = np.clip(np.exp(state["log_stiffness"])*opt["stiffness"],.05,50)
    extra = np.clip(np.exp(state["log_extra_damping"])*opt["damping"],1e-5,5)
    gain = np.clip(np.exp(state["log_fim_gain"]),1e-6,10)
    c0,ct = np.array(task["c0"]),np.array(task["c_target"])
    pose,target = np.array(task["pose0"]),np.array(task["pose_target"])
    tid = task["task_id"]
    rel = relative_log(pose,target)
    if entry["stage"] not in {"Final","Pose snapshot"}:
        rel = np.zeros(3)
    def ref(tau):
        phase = np.array([np.sin(2*math.pi*tau),np.cos(2*math.pi*tau),tau*(tid!=2)])
        features = np.r_[np.eye(3)[tid],c0,ct,rel*[20,20,2],phase]
        residual = meta["model_config"]["ref_scale"]*np.tanh(mlp(state,"net",features,3))
        base = (1-tau)*c0+tau*ct if tid==0 else c0
        envelope = 4*tau*(1-tau) if tid==0 else np.sin(math.pi*tau/2) if tid==1 else 1
        return K@(base+envelope*residual)
    q = np.array(task["metadata"]["rigid_loop_q_circle"])
    hard = closure["mode"]=="hard"
    if hard:
        q = chain.project(q,8)
    pi = np.zeros(chain.n); I = np.eye(chain.n)
    qs,poses,pis = [q],[pose],[pi]
    refs=[]; nus=[]; metrics={k:[] for k in ("hamiltonian","kinetic","potential","dissipation","closure_gap","speed","pose_error","balance_residual","fim_trace")}
    def energy(q,pi,qr):
        M = chain.assemble(q)[0]
        T = .5*pi@np.linalg.solve(M,pi)
        V = .5*np.sum(stiffness*(q-qr)**2)+chain.seam(q,closure)[0]
        return T+V,T,V
    h0 = energy(q,pi,ref(0))[0]; work=0.; loss=0.; diverged=False
    dt=1/opt["steps"]; n=max(opt["steps"],round(opt["steps"]*opt["horizon"]))
    for k in range(n):
        v = np.r_[pose,q,pi]
        if not np.isfinite(v).all() or np.max(np.abs(v))>1e6:
            diverged=True; break
        qr = ref(min(k/opt["steps"],1)); qrn = ref(min((k+1)/opt["steps"],1))
        M,D,F,gap,Jq = chain.assemble(q)
        nu = np.linalg.solve(M,pi)
        _,spring_seam,Dseam = chain.seam(q,closure)
        Dtotal = D.copy()
        Dtotal[3:,3:] += np.diag(extra)+Dseam+(gain*F if fim_on else 0)
        vx,vy,w = nu[:3]; px,py,_ = pi[:3]
        coad = np.array([-w*py,w*px,-px*vy+py*vx])
        spring = -stiffness*(q-qr)
        rhs = np.r_[coad,spring+spring_seam]
        pi_next = np.linalg.solve(I+dt*Dtotal@np.linalg.solve(M,I),pi+dt*rhs)
        nu_next = np.linalg.solve(M,pi_next)
        if hard:
            J = np.column_stack([np.zeros((2,3)),Jq])
            A = np.linalg.solve(M,J.T)
            nu_next -= A@np.linalg.solve(J@A+1e-8*np.eye(2),J@nu_next+10*gap)
            pi_next = M@nu_next
        xi = nu_next[:3]; th=xi[2]*dt; A,B=ab(th)
        inc = np.array([A*xi[0]*dt-B*xi[1]*dt,B*xi[0]*dt+A*xi[1]*dt,th])
        pose = compose(pose,inc)
        q_next = q+dt*nu_next[3:]
        if hard:
            q_next=chain.project(q_next,5)
            pi_next=chain.assemble(q_next)[0]@nu_next
        H,T,V=energy(q_next,pi_next,qrn)
        diss=nu_next@Dtotal@nu_next
        pref=-np.dot(stiffness*(q-qr),(qrn-qr)/dt)
        work+=dt*pref; loss+=dt*diss
        vals=[H,T,V,diss,np.linalg.norm(chain.kin(q_next)[0][-1]),np.linalg.norm(nu_next),
              np.linalg.norm(pose[:2]-target[:2]),H-h0-work+loss,
              np.trace(F)+chain.j*1e-10*max(1,np.max(np.abs(F)))]
        for key,value in zip(metrics,vals): metrics[key].append(value)
        q,pi=q_next,pi_next
        qs.append(q); poses.append(pose); pis.append(pi); nus.append(nu_next); refs.append(qr)
        if k%8==0: progress(k+1,n)
    if not refs: raise ValueError("Dynamics diverged before the first step")
    body=np.array([chain.kin(q)[0] for q in qs]); poses=np.array(poses); pis=np.array(pis)
    target_body=chain.kin(np.array(task["metadata"]["rigid_loop_q_target"]))[0]
    refs=np.array([chain.kin(q)[0] for q in refs])
    metrics["shape_error"]=np.sqrt(np.mean((body[1:]-target_body)**2,axis=(1,2)))
    obs=task["metadata"].get("rigid_loop_shared_observation_body")
    labels=[f"joint {i+1}" for i in range(chain.j)]+["pose x / body Px","pose y / body Py","pose θ / body Lz"]
    return serialize(body,poses,np.column_stack([qs,poses]),np.column_stack([pis[:,3:],pis[:,:3]]),labels,
                     target_body,target,metrics,dt,refs,np.array([transform(c,g) for c,g in zip(refs,poses[1:])]),
                     None if obs is None else transform(np.array(obs),target),diverged,
                     tolerance=closure["tolerance_rel"],fim_enabled=fim_on,
                     notes="Float64 WASM checkpoint dynamics. Joint momenta and body pose momenta follow articulated mechanics.")


def softmax(x):
    z=anp.exp(x-anp.max(x)); return z/anp.sum(z)


def spans(eta,cfg):
    return cfg["min_knot_span"]+(1-cfg["min_knot_span"]*len(eta))*softmax(eta)


def basis(eta,cfg):
    degree=cfg["degree"]; nc=cfg["n_ctrl"]
    knots=anp.concatenate([anp.zeros(degree+1),anp.cumsum(spans(eta,cfg))[:-1],anp.ones(degree+1)])
    u=anp.linspace(0,1,cfg["n_samples"])[:,None]
    kv=getval(knots)
    N=((u>=kv[:-1]) & (u<kv[1:])).astype(float)
    for k in range(1,degree+1):
        count=nc+degree-k
        ad=knots[k:k+count]-knots[:count]; bd=knots[k+1:k+1+count]-knots[1:1+count]
        am=anp.abs(getval(ad))>np.finfo(float).eps*16; bm=anp.abs(getval(bd))>np.finfo(float).eps*16
        a=(u-knots[:count])/anp.where(am,ad,1)*am
        b=(knots[k+1:k+1+count]-u)/anp.where(bm,bd,1)*bm
        N=a*N[:,:count]+b*N[:,1:count+1]
    endpoint=anp.eye(nc)[-1]
    return anp.where(u==1,endpoint,N[:,:nc])


def curve(P,eta,cfg):
    return anp.dot(basis(eta,cfg),P)


def project_controls(P,eta,cfg):
    if cfg["degree"]<1: return np.vstack([P[:-1],P[0]])
    s=spans(eta,cfg); ratio=s[-1]/max(s[0],1e-9)
    return np.vstack([P[:-2],P[0]-ratio*(P[1]-P[0]),P[0]])


def project_velocity(P,eta,vP,vE,cfg):
    V=vP.copy(); V[-1]=V[0]
    if cfg["degree"]<1: return V
    w=softmax(eta); slack=1-cfg["min_knot_span"]*len(eta); s=spans(eta,cfg)
    sf=max(s[0],1e-9); sl=s[-1]
    dsf=slack*w[0]*(np.eye(len(eta))[0]-w); dsl=slack*w[-1]*(np.eye(len(eta))[-1]-w)
    rdot=np.dot(dsl/sf-sl/(sf*sf)*dsf,vE)
    V[-2]=V[0]-sl/sf*(V[1]-V[0])-rdot*(P[1]-P[0])
    return V


def closure_energy(P,eta,cfg):
    gap=P[-1]-P[0]; s=spans(eta,cfg)
    d0=cfg["degree"]/anp.maximum(s[0],1e-8)*(P[1]-P[0])
    d1=cfg["degree"]/anp.maximum(s[-1],1e-8)*(P[-1]-P[-2])
    tan=d1/anp.maximum(anp.linalg.norm(d1),1e-8)-d0/anp.maximum(anp.linalg.norm(d0),1e-8)
    kp=cfg["closure_energy_scale"]/max(cfg["closure_tolerance"]**2,1e-8)
    return .5*kp*anp.sum(gap**2)+.5*cfg["tangent_closure_weight"]*kp*anp.sum(tan**2)


def sub(points,n):
    return points if n<=0 or len(points)<=n else points[np.rint(np.linspace(0,len(points)-1,n)).astype(int)]


def sinkhorn_cost(x,y,eps,n):
    C=anp.sum((x[:,None]-y[None,:])**2,axis=-1)
    f=anp.zeros(len(x)); g=anp.zeros(len(y))
    for _ in range(n):
        f=-eps*logsumexp((-C+g[None,:])/eps-math.log(len(y)),axis=1)
        g=-eps*logsumexp((-C+f[:,None])/eps-math.log(len(x)),axis=0)
    return anp.mean(f)+anp.mean(g)


def ot_state(P,eta,pose,obs,cfg,eps):
    x=sub(transform(curve(P,eta,cfg),pose),cfg["sinkhorn_force_source_points"])
    y=sub(obs,cfg["sinkhorn_force_target_points"]); n=cfg["sinkhorn_force_n_iter"]
    cross=sinkhorn_cost(x,y,eps,n)
    if cfg["sinkhorn_symmetric"]: cross=.5*(cross+sinkhorn_cost(y,x,eps,n))
    return cross-.5*sinkhorn_cost(x,x,eps,n)


def spline(state,meta,task,entry,opt,compared,progress):
    cfg=dict(meta["config"]); cfg["n_steps"]=opt["steps"]
    if not compared: cfg["closure_tolerance"]=opt["tolerance"]
    arrays={k:np.array(task[k]) for k in ("P0","eta0","P_target","eta_target","pose0","pose_target")}
    P0,E0,Pt,Et,G0,Gt=[arrays[k] for k in arrays]
    obs=None if task.get("observation_samples") is None else np.array(task["observation_samples"])
    mode=entry["mode"]; observation=entry["conditioning"]=="observation"
    context=np.zeros(64)
    if observation and obs is not None:
        h=mlp(state,"point_phi",obs,2,True); center=obs.mean(axis=0); X=obs-center
        stats=np.r_[center,np.mean(X[:,0]**2),np.mean(X[:,0]*X[:,1]),np.mean(X[:,1]**2)]
        context=mlp(state,"point_rho",np.r_[h.mean(axis=0),h.max(axis=0),stats],2,True)
    nc=cfg["n_ctrl"]; ne=len(E0); flatP=2*nc; block=flatP+ne+3
    def ref(tau):
        features=np.r_[np.eye(3)[{"open":0,"measurement":1,"periodic":2}[mode]],P0.ravel(),E0,G0,
                       np.zeros(block) if observation else np.r_[Pt.ravel(),Et,Gt],
                       np.eye(2)[int(observation)],context,[np.sin(2*math.pi*tau),np.cos(2*math.pi*tau),tau]]
        raw=mlp(state,"net",features,3); goal=np.tanh(raw[:block]); path=np.tanh(raw[block:])
        gp=goal[:flatP].reshape(nc,2); ge=goal[flatP:flatP+ne]; gg=goal[-3:]
        Pg=P0+1.8*gp if observation else Pt+.03*gp
        Eg=E0+2*ge if observation else Et+.12*ge
        Gg=G0+np.array([1.2,1.2,math.pi])*gg if observation else Gt+np.array([.03,.03,.06])*gg
        a=3*tau*tau-2*tau**3; env=4*tau*(1-tau)
        Pr=(1-a)*P0+a*Pg+env*.22*path[:flatP].reshape(nc,2)
        Er=(1-a)*E0+a*Eg+env*.9*path[flatP:flatP+ne]
        Gr=np.r_[G0[:2]+a*(Gg[:2]-G0[:2]),wrap(G0[2]+a*wrap(Gg[2]-G0[2]))]
        pathg=env*np.array([.2,.2,.35])*path[-3:]
        Gr=np.r_[Gr[:2]+pathg[:2],wrap(Gr[2]+pathg[2])]
        if mode=="periodic": Pr=project_controls(Pr,Er,cfg)
        return Pr,Er,Gr
    clamps={"kP":(.1,80),"kK":(.02,30),"kPoseXY":(.05,80),"kPoseTheta":(.02,40),
            "dP":(1e-4,10),"dK":(1e-4,10),"dPoseXY":(1e-4,15),"dPoseTheta":(1e-4,15)}
    pars={k:np.clip(np.exp(state["log_"+k])*opt["stiffness" if k.startswith("k") else "damping"],*v) for k,v in clamps.items()}
    kP,kE,kG,kTh,dP,dE,dG,dTh=[pars[k] for k in clamps]
    mP,mE,mG,mTh=[cfg[k] for k in ("mass_control","mass_knot","mass_pose_xy","mass_pose_theta")]
    P,E,G=P0.copy(),E0.copy(),G0.copy(); pP=np.zeros_like(P); pE=np.zeros_like(E); pG=np.zeros_like(G)
    Ps,Es,Gs,pPs,pEs,pGs=[P],[E],[G],[pP],[pE],[pG]
    refs=[]; refworld=[]; energies=[]; dissipation=[]; gaps=[]; ot=[]
    dt=1/opt["steps"]; n=max(opt["steps"],round(opt["steps"]*opt["horizon"]))
    seamgrad=value_and_grad(lambda P,E:closure_energy(P,E,cfg),argnum=(0,1))
    use_ot=cfg["use_sinkhorn_potential"] and cfg["sinkhorn_potential_weight"]!=0 and obs is not None
    y=sub(obs,cfg["sinkhorn_force_target_points"]) if obs is not None else None
    targetcost=sinkhorn_cost(y,y,cfg["sinkhorn_eps_end"],cfg["sinkhorn_force_n_iter"]) if obs is not None else 0.
    for i in range(n):
        tau=min(i/opt["steps"],1); Pr,Er,Gr=ref(tau)
        gradP=kP*(P-Pr); gradE=kE*(E-Er)
        diff=G-Gr; diff[2]=wrap(diff[2]); gradG=diff*np.array([kG,kG,kTh])
        Vot=0.
        if mode=="measurement":
            _,(a,b)=seamgrad(P,E); gradP+=a; gradE+=b
        if use_ot:
            eps=cfg["sinkhorn_eps_start"]*(1-tau)+cfg["sinkhorn_eps_end"]*tau
            fun=lambda P,E,G:cfg["sinkhorn_potential_weight"]*ot_state(P,E,G,obs,cfg,eps)
            Vot,(a,b,c)=value_and_grad(fun,argnum=(0,1,2))(P,E,G)
            gradP+=a; gradE+=b; gradG+=c
        pP=(pP-dt*gradP)/(1+dt*dP/mP); pE=(pE-dt*gradE)/(1+dt*dE/mE)
        pG=(pG-dt*gradG)/(1+dt*np.array([dG/mG,dG/mG,dTh/mTh]))
        P=P+dt*pP/mP; E=E+dt*pE/mE; G=G+dt*pG/np.array([mG,mG,mTh]); G[2]=wrap(G[2])
        if mode=="periodic":
            P=project_controls(P,E,cfg); pP=mP*project_velocity(P,E,pP/mP,pE/mE,cfg)
        T=.5*(np.sum(pP*pP)/mP+np.sum(pE*pE)/mE+np.sum(pG[:2]**2)/mG+pG[2]**2/mTh)
        Vs=.5*kP*np.sum((P-Pr)**2)+.5*kE*np.sum((E-Er)**2)
        Vg=.5*kG*np.sum((G[:2]-Gr[:2])**2)+.5*kTh*wrap(G[2]-Gr[2])**2
        Vc=closure_energy(P,E,cfg) if mode=="measurement" else 0.
        diss=dP*np.sum((pP/mP)**2)+dE*np.sum((pE/mE)**2)+dG*np.sum((pG[:2]/mG)**2)+dTh*(pG[2]/mTh)**2
        energies.append([T,Vs+Vg+Vot+Vc,T+Vs+Vg+Vot+Vc]); dissipation.append(diss); gaps.append(np.linalg.norm(P[-1]-P[0]))
        ot.append(max(0,ot_state(P,E,G,obs,cfg,cfg["sinkhorn_eps_end"])-.5*targetcost) if obs is not None else 0.)
        Ps.append(P); Es.append(E); Gs.append(G); pPs.append(pP); pEs.append(pE); pGs.append(pG)
        C=curve(Pr,Er,cfg); refs.append(C); refworld.append(transform(C,Gr))
        if i%4==0: progress(i+1,n)
    body=np.array([curve(P,E,cfg) for P,E in zip(Ps,Es)]); Gs=np.array(Gs)
    target=curve(Pt,Et,cfg)
    x=np.column_stack([np.array(Ps).reshape(n+1,-1),Es,Gs]); p=np.column_stack([np.array(pPs).reshape(n+1,-1),pEs,pGs])
    masses=np.r_[np.full(flatP,mP),np.full(ne,mE),mG,mG,mTh]; energies=np.array(energies)
    metrics={"kinetic":energies[:,0],"potential":energies[:,1],"hamiltonian":energies[:,2],"dissipation":dissipation,
             "closure_gap":gaps,"speed":np.linalg.norm(p[1:]/masses,axis=1),"pose_error":np.linalg.norm(Gs[1:,:2]-Gt[:2],axis=1),
             "shape_error":np.sqrt(np.mean((body[1:]-target)**2,axis=(1,2))),"sinkhorn":ot}
    labels=[f"P{i+1}.{axis}" for i in range(nc) for axis in ("x","y")]+[f"knot {i+1}" for i in range(ne)]+["pose x","pose y","pose θ"]
    return serialize(body,Gs,x,p,labels,target,Gt,metrics,dt,np.array(refs),np.array(refworld),obs,False,
                     tolerance=cfg["closure_tolerance"],
                     notes="Float64 WASM spline dynamics with checkpoint weights and differentiable Sinkhorn forces. FIM labels the saved measurement-closure variant.")


def serialize(body,poses,x,p,labels,target,target_pose,metrics,dt,refs,refworld,obs,diverged,**extra):
    times=np.arange(len(body))*dt
    result={"times":times,"metric_times":times[1:],"body":body,"world":np.array([transform(C,G) for C,G in zip(body,poses)]),
            "poses":poses,"phase_q":x,"phase_p":p,"coordinates":labels,"target_body":target,"target_world":transform(target,target_pose),
            "target_pose":target_pose,"reference_body":refs,"reference_world":refworld,"observations_world":obs,"metrics":metrics,
            "diverged":diverged,**extra}
    def plain(value):
        if isinstance(value,(np.ndarray,np.generic)):
            if not np.isfinite(value).all(): raise ValueError("Rollout contains nonfinite values; reduce stiffness or increase steps.")
            return value.tolist()
        if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [plain(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value): raise ValueError("Rollout contains nonfinite values")
        return value
    return plain(result)


class BrowserEngine:
    def __init__(self, manifest, root, progress=None):
        if manifest["schema_version"]!=2: raise ValueError("Unsupported checkpoint manifest")
        self.entries={e["id"]:e for e in manifest["checkpoints"]}
        self.root=Path(root); self.progress=progress or (lambda *args:None)

    @lru_cache(maxsize=4)
    def variant(self,filename):
        return json.loads(gzip.decompress((self.root/filename).read_bytes()))

    @lru_cache(maxsize=4)
    def weights(self,filename):
        with np.load(self.root/filename,allow_pickle=False) as arrays:
            return {k:arrays[k] for k in arrays.files}

    def run(self,data):
        if data.get("checkpoint") not in self.entries: raise ValueError("Unknown checkpoint")
        entry=self.entries[data["checkpoint"]]; opt=options(data,entry["defaults"])
        variant=self.variant(entry["variant_file"])
        group=data.get("group","hard_test" if entry["stage"]=="Hard transfer" else "test")
        if group not in variant["groups"]: raise ValueError("Unknown task group")
        index=number(data,"task_index",0,0,len(variant["groups"][group])-1,True)
        raw=variant["groups"][group][index]; entries=[entry]
        if data.get("compare"):
            other=self.entries.get(data["compare"])
            if other is None or other["family"]!=entry["family"]: raise ValueError("Comparison requires the same dynamics family")
            entries.append(other)
        runs=[]
        for j,e in enumerate(entries):
            t=copy.deepcopy(raw); meta=self.variant(e["variant_file"])["metadata"]
            old=np.array(t["pose_target"])
            t["pose_target"]=(old+np.array([opt["target_dx"],opt["target_dy"],math.radians(opt["target_angle"])] )).tolist()
            t["pose0"]=(np.array(t["pose0"])+[0,0,math.radians(opt["initial_angle"])]).tolist()
            if e["family"]=="spline" and t.get("observation_samples") is not None:
                body=(np.array(t["observation_samples"])-old[:2])@rotation(old[2])
                obs=transform(body,np.array(t["pose_target"]))
                if opt["noise"]: obs+=opt["noise"]*np.random.default_rng(opt["seed"]).standard_normal(obs.shape)
                t["observation_samples"]=obs.tolist()
            self.progress("checkpoint",j+1,len(entries))
            callback=lambda done,total:self.progress("steps",done,total)
            fn=spline if e["family"]=="spline" else multilink
            result=fn(self.weights(e["weights_file"]),meta,t,e,opt,j>0,callback)
            result.update(id=e["id"],label=e["label"],mode_label=e["mode_label"],conditioning=e["conditioning"],
                          checkpoint_epoch=e["checkpoint_epoch"],source_sha256=e["source_sha256"])
            runs.append(result)
        return {"schema_version":1,"runtime":"pyodide-numpy-wasm-float64","family":entry["family"],"group":group,
                "task_index":index,"configuration":opt,"runs":runs}
