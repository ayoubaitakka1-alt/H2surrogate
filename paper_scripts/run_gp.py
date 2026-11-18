import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/work/scripts')
from common import *
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as CK
from sklearn.model_selection import KFold
tr, va, sa = load_tables(); X, Xv, Xs = tr[FEATS].values, va[FEATS].values, sa[FEATS].values
t0=time.time()
def gp_fit_predict(Xa, ya, Xb):
    sx = StandardScaler().fit(Xa); k = CK(1.0,(1e-2,1e3))*RBF([1.0]*4,(1e-2,1e3)) + WhiteKernel(1e-2,(1e-8,1e-1))
    m = GPR(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(sx.transform(Xa), ya)
    return m.predict(sx.transform(Xb))
OUT={}
for tgt in ['T_peak','SOC_safe']:
    y, yv, ys = tr[tgt].values, va[tgt].values, sa[tgt].values
    oof=np.zeros(len(y)); f=[]
    for a,b in KFold(5,shuffle=True,random_state=42).split(X):
        p=gp_fit_predict(X[a],y[a],X[b]); oof[b]=p; f.append(metrics(y[b],p)['RMSE'])
    pv=gp_fit_predict(X,y,np.vstack([Xv,Xs])); p1,p2=pv[:len(Xv)],pv[len(Xv):]
    pred=np.full(len(y),np.nan)
    for L in sorted(tr.T_pre.unique())[1:-1]:
        te=(tr.T_pre==L).values; pred[te]=gp_fit_predict(X[~te],y[~te],X[te])
    m=~np.isnan(pred); lolo=metrics(y[m],pred[m])['RMSE']
    te=tr.T_pre.isin([tr.T_pre.min(),tr.T_pre.max()]).values; pe=gp_fit_predict(X[~te],y[~te],X[te])
    OUT[tgt]=dict(cv=float(np.mean(f)),off=metrics(yv,p1),conf=metrics(ys,p2),lolo=lolo,edge=metrics(y[te],pe)['RMSE'],pv=p1,pe=pe)
    print(tgt,{k:(v if not isinstance(v,dict) else round(v['RMSE'],3)) for k,v in OUT[tgt].items() if k not in('pv','pe')},round(time.time()-t0),flush=True)
# classification off-grid
pv=OUT['T_peak']['pv']; yv=va.T_peak.values
OUT['class']=dict(false_safe=int(((pv<=85)&(yv>85)).sum()),false_unsafe=int(((pv>85)&(yv<=85)).sum()))
print(OUT['class'], 'bias',OUT['T_peak']['off']['BIAS'], 'max',OUT['T_peak']['off']['MAXE'])
pickle.dump(OUT,open('/home/claude/work/results_gp.pkl','wb'))
