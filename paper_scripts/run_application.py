import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/work/scripts')
from common import *
import loader
from sklearn.preprocessing import StandardScaler

tr, va, sa = load_tables()
X, Xv = tr[FEATS].values, va[FEATS].values
M = pickle.load(open('/home/claude/work/results_main.pkl', 'rb'))
CP = loader.CP_H2
OUT = {}

# ---- surrogates used in the application (fitted on all 1134 training runs) ------------------
class Fitted:
    def __init__(self, name, tgt):
        self.name = name; y = tr[tgt].values
        self.sx = StandardScaler().fit(X); self.m = make(name)
        if name == 'MLP':
            self.sy = StandardScaler().fit(y.reshape(-1, 1)); self.m.fit(self.sx.transform(X), self.sy.transform(y.reshape(-1, 1)).ravel())
        else:
            self.m.fit(self.sx.transform(X), y)
    def predict(self, Z):
        p = self.m.predict(self.sx.transform(Z))
        return self.sy.inverse_transform(p.reshape(-1, 1)).ravel() if self.name == 'MLP' else p
svrT = Fitted('SVR (RBF)', 'T_peak'); mlpT = Fitted('MLP', 'T_peak'); mlpS = Fitted('MLP', 'SOC_safe')

# ---- fill time: deterministic function of the ramp rate ---------------------------------------
A = np.vstack([np.ones(len(tr)), 1 / tr.rate.values]).T
coef, *_ = np.linalg.lstsq(A, tr.t_end.values, rcond=None)
res = tr.t_end.values - A @ coef
OUT['fill_time'] = dict(a=coef[0], b=coef[1], max_abs_err=float(np.abs(res).max()))
t_fill = lambda r: coef[0] + coef[1] / np.asarray(r, float)
print('fill time fit: t = %.2f + %.1f/rate ; max abs error %.2f s' % (coef[0], coef[1], np.abs(res).max()))
va_res = va.t_end.values - (coef[0] + coef[1] / va.rate.values); print('validation fill-time max err', np.abs(va_res).max())

# ---- precooling electricity model -----------------------------------------------------------------
def cop(Tpre, Tamb, model):
    if model == 'const': return 1.5 * np.ones(np.broadcast(Tpre, Tamb).shape)
    Tc = np.asarray(Tpre, float) + 273.15; Th = np.asarray(Tamb, float) + 10.0 + 273.15
    return 0.4 * Tc / (Th - Tc)
def e_kwh(Tpre, Tamb, model):
    lift = np.maximum(np.asarray(Tamb, float) - np.asarray(Tpre, float), 0.0)
    return CP * lift / (3600.0 * cop(Tpre, Tamb, model))
# how good is the sensible-heat formula against the actual hose-temperature profile?
chk = []
for tank, f in [(3, 'train_type3.xlsx'), (4, 'train_type4.xlsx')]:
    runs = loader.load_training(DATA + f, tank)
    for r in runs[::7]:
        if r['T_pre'] <= r['T_amb'] - 20:
            d = r['run']; t = d.t.values
            Q_true = np.trapezoid(d.mdot.values / 1000 * CP * np.maximum(r['T_amb'] - d.T_hose.values, 0), t)
            m = np.trapezoid(d.mdot.values / 1000, t)
            Q_form = m * CP * (r['T_amb'] - r['T_pre'])
            chk.append(Q_true / Q_form)
OUT['duty_ratio'] = (float(np.mean(chk)), float(np.min(chk)), float(np.max(chk)), len(chk))
print('actual duty / sensible-heat formula: mean %.3f min %.3f max %.3f (n=%d)' % OUT['duty_ratio'])

# ---- setpoint optimisation -----------------------------------------------------------------------------
PRE = np.arange(-40, 40.01, 0.5); RATE = np.arange(10, 26.01, 0.5)
PG, RG = np.meshgrid(PRE, RATE, indexing='ij')
def optimise(tank, Tamb, margin=2.0, tmax=240.0, cop_model='carnot', surrogate=svrT):
    Z = np.column_stack([np.full(PG.size, float(tank == 4)), np.full(PG.size, Tamb), PG.ravel(), RG.ravel()])
    T = surrogate.predict(Z).reshape(PG.shape)
    ok = (T <= 85 - margin) & (t_fill(RG) <= tmax) & (PG <= Tamb)
    if not ok.any(): return None
    E = np.where(ok, e_kwh(PG, Tamb, cop_model), np.inf); emin = E.min()
    cand = ok & (E <= emin * 1.01 + 1e-12)
    tf = np.where(cand, t_fill(RG), np.inf); i = np.unravel_index(np.argmin(tf), tf.shape)
    return dict(Tpre=PG[i], rate=RG[i], t=float(t_fill(RG[i])), T=float(T[i]), e=float(e_kwh(PG[i], Tamb, cop_model)))

AMB = [20, 25, 30, 35, 40, 45, 50]
rows = []
for tank in (3, 4):
    for Ta in AMB:
        for cm in ('carnot', 'const'):
            s2 = optimise(tank, Ta, 2.0, 240.0, cm)
            e0 = float(e_kwh(-40.0, Ta, cm))
            # S0 : -40 C, 26 MPa/min (no weather or vehicle dependence)
            z0 = np.array([[float(tank == 4), Ta, -40.0, 26.0]])
            # S1 : no precooling, slowest ramp that meets 240 s
            r1 = float(np.ceil(coef[1] / (240.0 - coef[0]))); T1 = min(Ta, 40.0)
            z1 = np.array([[float(tank == 4), Ta, T1, r1]])
            rows.append(dict(tank=tank, T_amb=Ta, cop=cm, S2_Tpre=s2['Tpre'], S2_rate=s2['rate'], S2_t=s2['t'], S2_T=s2['T'],
                             S2_T_mlp=float(mlpT.predict(np.array([[float(tank == 4), Ta, s2['Tpre'], s2['rate']]]))[0]),
                             e_S0=e0, e_S2=s2['e'], saving=100 * (1 - s2['e'] / e0), t_S0=float(t_fill(26)),
                             S0_T=float(svrT.predict(z0)[0]), S1_rate=r1, S1_T=float(svrT.predict(z1)[0]), S1_SOC=float(mlpS.predict(z1)[0])))
SP = pd.DataFrame(rows); OUT['setpoints'] = SP
print(SP[SP.cop == 'carnot'].round(3).to_string())
print(SP[SP.cop == 'const'][['tank', 'T_amb', 'e_S0', 'e_S2', 'saving']].round(3).to_string())

# ---- sensitivity to fill-time limit and margin ------------------------------------------------------
TM = [170, 200, 240, 300, 360]; SENS = []
for tank in (3, 4):
    for margin in (2.0, 5.0):
        for tm in TM:
            for cm in ('carnot', 'const'):
                sv = []
                for Ta in AMB:
                    s2 = optimise(tank, Ta, margin, float(tm), cm)
                    e0 = float(e_kwh(-40.0, Ta, cm))
                    sv.append(100 * (1 - s2['e'] / e0) if s2 else np.nan)
                SENS.append(dict(tank=tank, margin=margin, tmax=tm, cop=cm, saving_mean=float(np.nanmean(sv)), saving_35=sv[3], n_infeasible=int(np.isnan(sv).sum())))
OUT['sens'] = pd.DataFrame(SENS); print(OUT['sens'][OUT['sens'].tmax.isin([170, 240, 360])].round(1).to_string())

# ---- Pareto data at 35 C -------------------------------------------------------------------------------------
PAR = {}
for tank in (3, 4):
    Z = np.column_stack([np.full(PG.size, float(tank == 4)), np.full(PG.size, 35.0), PG.ravel(), RG.ravel()])
    T = svrT.predict(Z).reshape(PG.shape); PAR[tank] = dict(T=T)
OUT['pareto'] = PAR

# ---- 85 degC contour from SVR vs. interpolation of the simulated grid ------------------------------------------
def crossing_interp(x, y, lim=85.0):
    # warmest T_pre where y crosses lim, by linear interpolation between grid levels (y increases with x)
    for i in range(len(x) - 1):
        if y[i] <= lim < y[i + 1]:
            return x[i] + (lim - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return np.nan
fine = np.arange(-40, 40.001, 0.01); CT = []
for tank in (3, 4):
    for Ta in AMB:
        for r in sorted(tr.rate.unique()):
            sub = tr[(tr.tank == tank) & (tr.T_amb == Ta) & (tr.rate == r)].sort_values('T_pre')
            xi = crossing_interp(sub.T_pre.values, sub.T_peak.values)
            Z = np.column_stack([np.full(len(fine), float(tank == 4)), np.full(len(fine), Ta), fine, np.full(len(fine), r)])
            T = svrT.predict(Z); j = np.where(T <= 85)[0]
            xs = fine[j.max()] if len(j) else np.nan
            CT.append(dict(tank=tank, T_amb=Ta, rate=r, interp=xi, svr=xs))
CT = pd.DataFrame(CT); CT['diff'] = (CT.svr - CT.interp).abs(); OUT['contour'] = CT
print(CT.groupby('tank').agg(svr_min=('svr', 'min'), svr_max=('svr', 'max'), int_min=('interp', 'min'), int_max=('interp', 'max'), mean_diff=('diff', 'mean'), max_diff=('diff', 'max'), n_nan=('interp', lambda s: s.isna().sum())))

# ---- decision test on independent simulations ------------------------------------------------------------------------
def decide(pred_T, d, margin=2.0, tmax=240.0, cop_model='carnot'):
    out = []
    for (tank, Ta), g in d.groupby(['tank', 'T_amb']):
        idx = g.index.values
        ok = (pred_T[idx] <= 85 - margin) & (t_fill(d.loc[idx, 'rate'].values) <= tmax) & (d.loc[idx, 'T_pre'].values <= Ta)
        ok_true = (d.loc[idx, 'T_peak'].values <= 85 - margin) & (t_fill(d.loc[idx, 'rate'].values) <= tmax) & (d.loc[idx, 'T_pre'].values <= Ta)
        e = e_kwh(d.loc[idx, 'T_pre'].values, Ta, cop_model)
        def pick(mask):
            if not mask.any(): return None
            Em = np.where(mask, e, np.inf); cand = mask & (Em <= Em.min() * 1.01 + 1e-12)
            tf = np.where(cand, t_fill(d.loc[idx, 'rate'].values), np.inf); return int(np.argmin(tf))
        k, kt = pick(ok), pick(ok_true)
        out.append(dict(tank=tank, T_amb=Ta, chosen=None if k is None else idx[k], best=None if kt is None else idx[kt],
                        true_T=np.nan if k is None else d.loc[idx[k], 'T_peak'],
                        e_chosen=np.nan if k is None else e[k], e_best=np.nan if kt is None else e[kt]))
    return pd.DataFrame(out)
DEC = {}
for name in MODELS:
    D = decide(M['C'][('T_peak', name)]['pred'], va)
    D['unsafe'] = D.true_T >= 85; D['over_margin'] = D.true_T > 83; D['same'] = D.chosen == D.best
    DEC[name] = D
OUT['decision'] = DEC
for name in MODELS:
    D = DEC[name]; print(f"{name:18s} cases {len(D)} none-feasible {D.chosen.isna().sum()} unsafe(>=85) {int(D.unsafe.sum())} >83: {int(D.over_margin.sum())} same-as-best {int(D.same.sum())} max true T {np.nanmax(D.true_T):.2f}")
pickle.dump(OUT, open('/home/claude/work/results_app.pkl', 'wb'))
