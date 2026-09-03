"""Scaling sweep for the TN-encoder conditioning ablation: n in {10,12,14}, fixed
p=4, multi-seed, paired bootstrap stats. Saves ablation_scaling.json and
diffusion_qaoa_scaling.png. Consistent (moderate) optimizer budget across sizes so
absolute ratios are comparable; the trend across n is the object of interest."""
import numpy as np, torch, torch.nn as nn, networkx as nx, json, time, functools, warnings
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# consistent optimizer budget across all sizes
DTS, MAXIT = (0.6, 0.9, 1.2), 150

def edges_of(G): return [(i, j) for i, j in G.edges()]
def bit_matrix(n):
    xs = np.arange(2**n); return ((xs[:,None] >> (n-1-np.arange(n))[None,:]) & 1).astype(np.int8)
def cut_values(edges, n, BITS):
    C = np.zeros(2**n)
    for i,j in edges: C += (BITS[:,i] != BITS[:,j])
    return C
def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j*np.sin(beta); psi = psi.reshape([2]*n)
    for q in range(n):
        psi = np.moveaxis(psi, q, 0); a,b = psi[0].copy(), psi[1].copy()
        psi[0] = c*a+s*b; psi[1] = s*a+c*b; psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)
def qaoa_energy(params, C, n, p):
    g, be = params[:p], params[p:]; psi = np.ones(2**n, complex)/np.sqrt(2**n)
    for l in range(p):
        psi = np.exp(-1j*g[l]*C)*psi; psi = apply_mixer(psi, be[l], n)
    return float(np.dot(np.abs(psi)**2, C))
def tqa_init(p, dt):
    l = np.arange(1,p+1); frac = (l-0.5)/p
    return np.concatenate([frac*dt, (1-frac)*dt])
def canonicalize(params, p):
    g = np.array(params[:p],float); b = np.array(params[p:],float)
    if g[0]<0: g,b = -g,-b
    g = np.mod(g,2*np.pi); g = np.where(g>np.pi,g-2*np.pi,g)
    b = np.mod(b,np.pi);   b = np.where(b>np.pi/2,b-np.pi,b)
    return np.concatenate([g,b]).astype(np.float32)
def optimize_qaoa(C, n, p):
    bv, bx = -np.inf, None
    for dt in DTS:
        r = minimize(lambda x: -qaoa_energy(x,C,n,p), tqa_init(p,dt), method="COBYLA", options={"maxiter":MAXIT})
        if -r.fun > bv: bv, bx = -r.fun, r.x
    return canonicalize(bx,p), bv
def spectrum_embedding(G, n):
    L = nx.normalized_laplacian_matrix(G).toarray(); eig = np.sort(np.linalg.eigvalsh(L))
    return np.concatenate([eig,[G.number_of_edges()/(n*(n-1)/2)]]).astype(np.float32)
def random_graph_mixed(n, rng):
    while True:
        k = rng.integers(3)
        if   k==0: G = nx.erdos_renyi_graph(n, rng.uniform(0.3,0.6), seed=int(rng.integers(1<<30)))
        elif k==1: G = nx.random_regular_graph(int(rng.choice([3,4,5])), n, seed=int(rng.integers(1<<30)))
        else:      G = nx.barabasi_albert_graph(n, int(rng.choice([2,3])), seed=int(rng.integers(1<<30)))
        if nx.is_connected(G) and G.number_of_edges()>0: return G
F = 4
def vertex_leaves(G, n, n_leaves):
    deg=dict(G.degree()); clus=nx.clustering(G); tri=nx.triangles(G); aND=nx.average_neighbor_degree(G)
    nodes=list(G.nodes())
    feats={v:[deg[v]/(n-1), clus[v], aND[v]/(n-1), tri[v]/max(1,(n-1)*(n-2)/2)] for v in nodes}
    try: fied=np.asarray(nx.fiedler_vector(G, method="tracemin_lu"))
    except Exception: fied=np.zeros(len(nodes))
    if fied[int(np.argmax(np.abs(fied)))]<0: fied=-fied
    order=sorted(range(len(nodes)), key=lambda i:(-feats[nodes[i]][0], -fied[i]))
    leaves=np.array([feats[nodes[i]] for i in order], np.float32)
    return np.concatenate([leaves, np.zeros((n_leaves-len(leaves),F),np.float32)],0)

T = 100
betas_dif = np.linspace(1e-4,0.02,T).astype(np.float32); alphas=1-betas_dif; abar=np.cumprod(alphas)
sqrt_abar=torch.tensor(np.sqrt(abar)); sqrt_1mabar=torch.tensor(np.sqrt(1-abar))
def tstep(t,d=32):
    half=d//2; fr=torch.exp(-np.log(10000)*torch.arange(half)/half); a=t[:,None].float()*fr[None]
    return torch.cat([torch.sin(a),torch.cos(a)],-1)
class Denoiser(nn.Module):
    def __init__(self, dim, cond_dim, tdim=32, h=128):
        super().__init__(); self.tdim=tdim
        self.net=nn.Sequential(nn.Linear(dim+cond_dim+tdim,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,dim))
    def forward(self,x,t,c): return self.net(torch.cat([x,c,tstep(t,self.tdim)],-1))
CHI, COND_TN = 6, 8
class TTNEncoder(nn.Module):
    def __init__(self, n_leaves):
        super().__init__(); self.embed=nn.Linear(F,CHI)
        self.Ws=nn.ParameterList([nn.Parameter(torch.randn(CHI,CHI,CHI)*0.1) for _ in range(int(np.log2(n_leaves)))])
        self.head=nn.Linear(CHI,COND_TN); self.ln=nn.LayerNorm(COND_TN)
    def forward(self, leaves):
        x=self.embed(torch.tanh(leaves))
        for W in self.Ws:
            B,L,chi=x.shape; x=x.view(B,L//2,2,chi)
            x=torch.einsum('apq,blp,blq->bla',W,x[:,:,0,:],x[:,:,1,:]); x=x/(x.norm(dim=-1,keepdim=True)+1e-6)
        return self.ln(self.head(x.squeeze(1)))
class MLPEncoder(nn.Module):
    def __init__(self, n_leaves, h=64):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(n_leaves*F,h),nn.SiLU(),nn.Linear(h,COND_TN)); self.ln=nn.LayerNorm(COND_TN)
    def forward(self, leaves): return self.ln(self.net(torch.tanh(leaves)))

def build_size(n, p, n_train, n_test, rng):
    n_leaves = 1 << int(np.ceil(np.log2(n))); BITS = bit_matrix(n)
    X,SP,LV=[],[],[]
    for gi in range(n_train):
        if gi%40==0: print(f"    [n={n}] train {gi}/{n_train}")
        G=random_graph_mixed(n,rng); C=cut_values(edges_of(G),n,BITS); x,_=optimize_qaoa(C,n,p)
        X.append(x); SP.append(spectrum_embedding(G,n)); LV.append(vertex_leaves(G,n,n_leaves))
    teSP,teLV,teC,teOPT=[],[],[],[]
    for gi in range(n_test):
        G=random_graph_mixed(n,rng); C=cut_values(edges_of(G),n,BITS)
        teSP.append(spectrum_embedding(G,n)); teLV.append(vertex_leaves(G,n,n_leaves))
        teC.append(C); teOPT.append(C.max())
    tr=dict(X=np.array(X,np.float32),SP=np.array(SP,np.float32),LV=np.array(LV,np.float32))
    te=dict(SP=np.array(teSP,np.float32),LV=np.array(teLV,np.float32),C=teC,OPT=np.array(teOPT))
    return tr, te, n_leaves

@torch.no_grad()
def sample_cond(deno, dim, cond, n):
    cond=cond.repeat(n,1); x=torch.randn(n,dim)
    for ti in reversed(range(T)):
        t=torch.full((n,),ti); pred=deno(x,t,cond); a,ab,b=alphas[ti],abar[ti],betas_dif[ti]
        x=(1/np.sqrt(a))*(x-(1-a)/np.sqrt(1-ab)*pred)+(np.sqrt(b)*torch.randn_like(x) if ti>0 else 0.0)
    return x.numpy()

def run_seed(n,p,tr,te,n_leaves,seed,K=8):
    torch.manual_seed(seed); dim=2*p
    Xm,Xs=tr["X"].mean(0),tr["X"].std(0)+1e-6; SPm,SPs=tr["SP"].mean(0),tr["SP"].std(0)+1e-6
    Xa_t=torch.tensor((tr["X"]-Xm)/Xs); SPa_t=torch.tensor((tr["SP"]-SPm)/SPs)
    SPte_t=torch.tensor((te["SP"]-SPm)/SPs); LVa_t=torch.tensor(tr["LV"]); LVte_t=torch.tensor(te["LV"])
    def train(cond_dim,cond_fn,enc=None,steps=3000):
        deno=Denoiser(dim,cond_dim); ps=list(deno.parameters())+([] if enc is None else list(enc.parameters()))
        opt=torch.optim.Adam(ps,1e-3)
        for _ in range(steps):
            idx=torch.randint(0,len(Xa_t),(128,)); x0=Xa_t[idx]; cond=cond_fn(idx,enc)
            t=torch.randint(0,T,(128,)); noise=torch.randn_like(x0)
            xt=sqrt_abar[t][:,None]*x0+sqrt_1mabar[t][:,None]*noise
            loss=((deno(xt,t,cond)-noise)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        return deno,enc
    M={}
    M["spectrum"]=train(tr["SP"].shape[1], lambda idx,e: SPa_t[idx])
    M["TTN"]=train(COND_TN, lambda idx,e: e(LVa_t[idx]), TTNEncoder(n_leaves))
    M["MLP"]=train(COND_TN, lambda idx,e: e(LVa_t[idx]), MLPEncoder(n_leaves))
    def cond_te(name,g):
        if name=="spectrum": return SPte_t[g:g+1]
        with torch.no_grad(): return M[name][1](LVte_t[g:g+1])
    r={m:{"one":[],"bk":[]} for m in M}
    for g in range(len(te["C"])):
        C,optv=te["C"][g],te["OPT"][g]
        for m in M:
            xs=sample_cond(M[m][0],dim,cond_te(m,g),K)*Xs+Xm
            es=[qaoa_energy(x,C,n,p) for x in xs]
            r[m]["one"].append(es[0]/optv); r[m]["bk"].append(max(es)/optv)
    return {m:{k:np.array(v) for k,v in r[m].items()} for m in r}

def paired(a,b,rng,B=5000):
    d=a-b; n=len(d); idx=rng.integers(0,n,(B,n)); boot=d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5)), float(np.mean(d>0))

def sweep():
    sizes=[(10,4),(12,4),(14,4)]; N_TRAIN,N_TEST,SEEDS=200,80,3
    brng=np.random.default_rng(0); res={}
    for (n,p) in sizes:
        t0=time.time(); print(f"\n===== n={n}, p={p} =====")
        rng=np.random.default_rng(2024+n)
        tr,te,nl=build_size(n,p,N_TRAIN,N_TEST,rng)
        per={m:{"one":[],"bk":[]} for m in ["spectrum","TTN","MLP"]}
        for s in range(SEEDS):
            rr=run_seed(n,p,tr,te,nl,seed=7000+s)
            for m in per: per[m]["one"].append(rr[m]["one"]); per[m]["bk"].append(rr[m]["bk"])
        for m in per:
            per[m]["one"]=np.array(per[m]["one"]); per[m]["bk"]=np.array(per[m]["bk"])  # [seed,graph]
        one_mg={m:per[m]["one"].mean(0) for m in per}; bk_mg={m:per[m]["bk"].mean(0) for m in per}
        entry={"n":n,"p":p}
        for m in per:
            entry[m]={"one_mean":float(per[m]["one"].mean()), "one_seedstd":float(per[m]["one"].mean(1).std()),
                      "bk_mean":float(per[m]["bk"].mean()), "bk_seedstd":float(per[m]["bk"].mean(1).std()),
                      "one_graph_se":float(one_mg[m].std()/np.sqrt(len(one_mg[m])))}
        for pair in [("TTN","spectrum"),("MLP","spectrum"),("TTN","MLP")]:
            d,lo,hi,wr=paired(one_mg[pair[0]],one_mg[pair[1]],brng)
            entry[f"{pair[0]}-{pair[1]}"]={"diff":d,"lo":lo,"hi":hi,"win":wr}
        res[n]=entry
        print(f"  n={n} 1-shot: "+", ".join(f"{m} {entry[m]['one_mean']:.3f}" for m in per))
        print(f"  TTN-spectrum {entry['TTN-spectrum']['diff']:+.4f} CI[{entry['TTN-spectrum']['lo']:+.4f},{entry['TTN-spectrum']['hi']:+.4f}]  "
              f"TTN-MLP {entry['TTN-MLP']['diff']:+.4f} CI[{entry['TTN-MLP']['lo']:+.4f},{entry['TTN-MLP']['hi']:+.4f}]")
        print(f"  n={n} done in {time.time()-t0:.0f}s")
        json.dump(res, open("ablation_scaling.json","w"), indent=1)  # checkpoint each size
    return res

def make_figure(res):
    ns=sorted(res); cols={"spectrum":"#c94","TTN":"#284","MLP":"#39c"}
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for m in ["spectrum","TTN","MLP"]:
        y=[res[n][m]["one_mean"] for n in ns]; e=[res[n][m]["one_seedstd"] for n in ns]
        ax[0].errorbar(ns,y,yerr=e,marker="o",capsize=4,color=cols[m],label=m,lw=2)
        yb=[res[n][m]["bk_mean"] for n in ns]
        ax[0].plot(ns,yb,marker="s",ls="--",color=cols[m],alpha=.5)
    ax[0].set_xlabel("graph size $n$ (qubits)"); ax[0].set_ylabel("approx. ratio")
    ax[0].set_title("Conditioning vs size  (solid: 1-shot, dashed: best-of-8)")
    ax[0].set_xticks(ns); ax[0].grid(alpha=.3); ax[0].legend()
    for pair,c in [("TTN-spectrum","#284"),("TTN-MLP","#a24")]:
        d=[res[n][pair]["diff"] for n in ns]; lo=[res[n][pair]["lo"] for n in ns]; hi=[res[n][pair]["hi"] for n in ns]
        ax[1].plot(ns,d,marker="o",color=c,lw=2,label=pair)
        ax[1].fill_between(ns,lo,hi,color=c,alpha=.15)
    ax[1].axhline(0,color="k",lw=1,ls=":")
    ax[1].set_xlabel("graph size $n$ (qubits)"); ax[1].set_ylabel("paired 1-shot difference")
    ax[1].set_title("TN advantage vs size (bootstrap 95% CI)"); ax[1].set_xticks(ns); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("diffusion_qaoa_scaling.png",dpi=150,bbox_inches="tight")
    print("saved diffusion_qaoa_scaling.png")

if __name__=="__main__":
    res=sweep(); make_figure(res); print("SWEEP DONE")
