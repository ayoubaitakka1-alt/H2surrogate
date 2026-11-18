import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/work/scripts')
from common import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

tr, va, sa = load_tables()
X, Xv, Xs = tr[FEATS].values, va[FEATS].values, sa[FEATS].values
t00 = time.time()
def log(*a): print(f'[{time.time()-t00:6.0f}s]', *a, flush=True)
M = pickle.load(open('/home/claude/work/results_main.pkl', 'rb'))
OUT = {}

# ---- exceedance classification -------------------------------------------------------------
def cls_counts(ytrue, ypred, lim=85.0):
    t = ytrue >= lim; p = ypred >= lim
    return dict(n=len(t), err=int((t != p).sum()), false_safe=int((t & ~p).sum()), false_unsafe=int((~t & p).sum()), n_exceed=int(t.sum()))
CL = {}
for name in MODELS:
    CL[(name, 'random')] = cls_counts(tr.T_peak.values, M['A'][('T_peak', name)]['oof'])
    CL[(name, 'offgrid')] = cls_counts(va.T_peak.values, M['C'][('T_peak', name)]['pred'])
    CL[(name, 'setA')] = cls_counts(sa.T_peak.values, M['S'][('T_peak', name)]['pred'])
    for fac in ['T_amb', 'T_pre', 'rate']:
        d = M['D'][('T_peak', name, fac)]
        CL[(name, 'edge_' + fac)] = cls_counts(tr.T_peak.values[d['idx']], d['pred'])
    # LOLO: pooled over the three factors
    n = e = fs = fu = ne = 0
    for fac in ['T_amb', 'T_pre', 'rate']:
        p = M['B'][('T_peak', name, fac)]['pred']; m = ~np.isnan(p)
        c = cls_counts(tr.T_peak.values[m], p[m]); n += c['n']; e += c['err']; fs += c['false_safe']; fu += c['false_unsafe']; ne += c['n_exceed']
    CL[(name, 'lolo')] = dict(n=n, err=e, false_safe=fs, false_unsafe=fu, n_exceed=ne)
# direct classifiers trained on the 1134 runs (off-grid test and SET A)
yc = (tr.T_peak.values >= 85).astype(int)
sx = StandardScaler().fit(X)
clfs = {'Logistic': LogisticRegression(C=10, max_iter=5000), 'SVC (RBF)': SVC(C=100, gamma='scale'),
        'XGBoost clf': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
        'MLP clf': MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=0.01, learning_rate_init=0.005, max_iter=5000, random_state=42)}
for nm, c in clfs.items():
    Xa = X if 'XGB' in nm else sx.transform(X)
    c.fit(Xa, yc)
    for key, XX, d in [('offgrid', Xv, va), ('setA', Xs, sa)]:
        pr = c.predict(XX if 'XGB' in nm else sx.transform(XX)).astype(bool)
        t = d.T_peak.values >= 85
        CL[(nm, key)] = dict(n=len(t), err=int((t != pr).sum()), false_safe=int((t & ~pr).sum()), false_unsafe=int((~t & pr).sum()), n_exceed=int(t.sum()))
OUT['CL'] = CL
log('classification done')

# distance to limit of the misclassified runs
mis = {}
for name in MODELS:
    p = M['C'][('T_peak', name)]['pred']; t = va.T_peak.values
    w = (t >= 85) != (p >= 85)
    mis[name] = dict(true=t[w], pred=p[w], tank=va.tank.values[w], amb=va.T_amb.values[w], pre=va.T_pre.values[w], rate=va.rate.values[w])
OUT['MIS'] = mis

# ---- learning curves (off-grid RMSE vs. number of training runs) ---------------------------
LC = {}
rng = np.random.RandomState(0)
sizes = [50, 100, 200, 400, 800]
for tgt in ['T_peak', 'SOC_safe']:
    y, yv = tr[tgt].values, va[tgt].values
    for name in MODELS:
        for n in sizes:
            r = []
            for d in range(5):
                idx = np.random.RandomState(100 * d + n).choice(len(y), n, replace=False)
                r.append(metrics(yv, fit_predict(name, X[idx], y[idx], Xv))['RMSE'])
            LC[(tgt, name, n)] = (float(np.mean(r)), float(np.min(r)), float(np.max(r)))
        LC[(tgt, name, 1134)] = (M['C'][(tgt, name)]['m']['RMSE'],) * 3
    log('learning curve done', tgt)
OUT['LC'] = LC

# ---- permutation importance on the validation set ---------------------------------------------
PI = {}
names = ['tank', 'T_amb', 'T_pre', 'rate']
for tgt in ['T_peak', 'SOC_safe', 'P_safe']:
    y, yv = tr[tgt].values, va[tgt].values
    for name in ['SVR (RBF)', 'MLP', 'XGBoost', 'Random Forest']:
        sc = StandardScaler().fit(X)
        # fit once, predict many times
        m = make(name)
        if name in ('SVR (RBF)', 'MLP'):
            if name == 'MLP':
                sy = StandardScaler().fit(y.reshape(-1, 1)); m.fit(sc.transform(X), sy.transform(y.reshape(-1, 1)).ravel())
                pred = lambda Z: sy.inverse_transform(m.predict(sc.transform(Z)).reshape(-1, 1)).ravel()
            else:
                m.fit(sc.transform(X), y); pred = lambda Z: m.predict(sc.transform(Z))
        else:
            m.fit(X, y); pred = lambda Z: m.predict(Z)
        base = np.sqrt(np.mean((pred(Xv) - yv)**2))
        rs = np.random.RandomState(1)
        for j, fn in enumerate(names):
            inc = []
            for k in range(30):
                Z = Xv.copy(); Z[:, j] = rs.permutation(Z[:, j]); inc.append(np.sqrt(np.mean((pred(Z) - yv)**2)) - base)
            PI[(tgt, name, fn)] = (float(np.mean(inc)), float(np.std(inc)))
    log('perm importance done', tgt)
OUT['PI'] = PI
pickle.dump(OUT, open('/home/claude/work/results_extras.pkl', 'wb'))
log('saved extras')
