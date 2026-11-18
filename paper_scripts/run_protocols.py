import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/work/scripts')
from sklearn.model_selection import KFold
from common import *

tr, va, sa = load_tables()
X, Xv, Xs = tr[FEATS].values, va[FEATS].values, sa[FEATS].values
TARGETS = ['T_peak', 'SOC_safe', 'P_safe', 'TL_peak', 'SOC_end']
OUT = {}
t00 = time.time()
def log(*a):
    print(f'[{time.time()-t00:6.0f}s]', *a, flush=True)

log('check T_peak==T_end:', (tr.T_peak - tr.T_end).abs().max(), (va.T_peak - va.T_end).abs().max(), (sa.T_peak - sa.T_end).abs().max())

# ---- A. random 5-fold CV ------------------------------------------------------------------
A = {}
for tgt in TARGETS:
    y = tr[tgt].values
    for name in MODELS:
        folds = []; oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=42).split(X):
            p = fit_predict(name, X[a], y[a], X[b]); oof[b] = p; folds.append(metrics(y[b], p))
        df = pd.DataFrame(folds)
        A[(tgt, name)] = dict(mean=df.mean().to_dict(), std=df.std(ddof=0).to_dict(), oof=oof)
    log('A done', tgt)
OUT['A'] = A

# ---- C. independent off-grid validation (223) + SET A confirmation (16, Type 3) -----------
C = {}; S = {}
for tgt in TARGETS:
    y, yv, ys = tr[tgt].values, va[tgt].values, sa[tgt].values
    for name in MODELS:
        pv = fit_predict(name, X, y, np.vstack([Xv, Xs]))
        p1, p2 = pv[:len(Xv)], pv[len(Xv):]
        C[(tgt, name)] = dict(m=metrics(yv, p1), pred=p1,
                              m_in=metrics(yv[va.T_amb <= 47], p1[va.T_amb <= 47]),
                              m_out=metrics(yv[va.T_amb > 47], p1[va.T_amb > 47]),
                              m_t3=metrics(yv[va.tank == 3], p1[va.tank == 3]), m_t4=metrics(yv[va.tank == 4], p1[va.tank == 4]))
        S[(tgt, name)] = dict(m=metrics(ys, p2), pred=p2)
    log('C done', tgt)
OUT['C'] = C; OUT['S'] = S

# ---- B. leave-one-level-out (interior levels) ---------------------------------------------
B = {}
levels = {'T_amb': sorted(tr.T_amb.unique())[1:-1], 'T_pre': sorted(tr.T_pre.unique())[1:-1], 'rate': sorted(tr.rate.unique())[1:-1]}
for tgt in TARGETS:
    y = tr[tgt].values
    for name in MODELS:
        for fac, lv in levels.items():
            pred = np.full(len(y), np.nan)
            for L in lv:
                te = (tr[fac] == L).values
                pred[te] = fit_predict(name, X[~te], y[~te], X[te])
            msk = ~np.isnan(pred)
            B[(tgt, name, fac)] = dict(m=metrics(y[msk], pred[msk]), pred=pred)
    log('B done', tgt)
OUT['B'] = B

# ---- D. edge extrapolation (withhold the two extreme levels of a factor) -------------------
D = {}
for tgt in TARGETS:
    y = tr[tgt].values
    for name in MODELS:
        for fac in ['T_amb', 'T_pre', 'rate']:
            lo, hi = tr[fac].min(), tr[fac].max()
            te = tr[fac].isin([lo, hi]).values
            p = fit_predict(name, X[~te], y[~te], X[te])
            D[(tgt, name, fac)] = dict(m=metrics(y[te], p), pred=p, idx=np.where(te)[0])
    log('D done', tgt)
OUT['D'] = D

# ---- MLP seed spread (off-grid) -------------------------------------------------------------
SEEDS = {}
for tgt in TARGETS:
    y, yv = tr[tgt].values, va[tgt].values
    r = []
    for s in range(10):
        p = fit_predict('MLP', X, y, Xv, seed=s); r.append(metrics(yv, p))
    SEEDS[tgt] = pd.DataFrame(r)
    log('seeds done', tgt)
OUT['SEEDS'] = SEEDS

pickle.dump(OUT, open('/home/claude/work/results_main.pkl', 'wb'))
log('saved main results')
