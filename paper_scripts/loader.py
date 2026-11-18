"""Load H2FillS training and validation time series, split into runs, extract per-run features."""
import numpy as np, pandas as pd

COLS = {
 'Time [s]':'t','Mass flow [g/s]':'mdot','Hose (breakaway) press [MPa]':'P_hose','Hose (breakaway) temp [degC]':'T_hose',
 'Vehicle tank #0 gas press [MPa]':'P','Vehicle tank #0 gas temp [degC]':'T',
 'Vehicle tank #0 liner surface temp [degC]':'T_liner','Vehicle tank #0 SOC [-]':'SOC',
 'Ambient temp [degC]':'T_amb','Vehicle tank #0 inlet gas temp [degC]':'T_in',
}

def split_runs(df):
    t = df['t'].values
    starts = [0] + [i for i in range(1, len(t)) if t[i] < t[i-1]]
    ends = starts[1:] + [len(t)]
    return [df.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]

def read_results(path):
    d = pd.read_excel(path, sheet_name='Results')
    d = d.rename(columns=COLS)[list(COLS.values())]
    return d

def infer_conditions(run):
    amb = float(run['T_amb'].iloc[0])
    # fuel delivery temperature: hose temp once the 30 s transient is over
    late = run[run['t'] >= 30]
    pre = float(late['T_hose'].iloc[0]) if len(late) else np.nan
    # ramp: initial slope of hose pressure
    rate = (run['P_hose'].iloc[1] - run['P_hose'].iloc[0]) / (run['t'].iloc[1] - run['t'].iloc[0]) * 60
    return amb, pre, rate

def load_training(path, tank):
    runs = split_runs(read_results(path))
    out = []
    for k, r in enumerate(runs):
        amb, pre, rate = infer_conditions(r)
        out.append(dict(tank=tank, T_amb=amb, T_pre=round(pre, 1), rate=round(rate, 2), run=r, idx=k))
    return out

def load_validation(path, tank):
    runs = split_runs(read_results(path))
    par = pd.read_excel(path, sheet_name='Parameters')
    par.columns = ['grp', 'src', 'comp', 'param', 'value']
    fc = par[par.comp == 'Fill Condition']
    keys = list(dict.fromkeys(zip(fc.grp, fc.src)))
    out = []
    # map each run to its Parameters block, in order
    for k, (r, key) in enumerate(zip(runs, keys)):
        blk = fc[(fc.grp == key[0]) & (fc.src == key[1])].set_index('param')['value']
        out.append(dict(tank=tank,
                        T_amb=float(blk['Ambient Temperature [degC]']),
                        T_pre=float(blk['Fuel Delivery Temperature [degC]']),
                        rate=float(blk['Pressure Ramp Rate [MPa/min]']),
                        run=r, idx=k, grp=key[0], src=key[1]))
    return out, len(runs), len(keys)


CP_H2 = 14.3  # kJ/kg/K, as used in both manuscripts
LIMIT = 85.0

def run_features(run, T_pre, T_amb, limit=LIMIT):
    t = run['t'].values.astype(float); T = run['T'].values.astype(float)
    P = run['P'].values.astype(float); S = run['SOC'].values.astype(float)
    TL = run['T_liner'].values.astype(float)
    md = run['mdot'].values.astype(float)  # g/s
    f = dict(T_end=T[-1], SOC_end=S[-1], P_end=P[-1], t_end=t[-1], T_peak=T.max(),
             TL_peak=TL.max(), monotone=bool(np.all(np.diff(T) >= -1e-6)))
    # mass dispensed [kg] and thermal duty [kJ]
    m_kg = np.trapezoid(md, t) / 1000.0
    f['m_H2'] = m_kg
    f['Q_pre'] = np.trapezoid(md / 1000.0 * CP_H2 * (T_amb - T_pre), t)
    # first crossing of the limit, with linear interpolation inside the step
    idx = np.where(T >= limit)[0]
    if len(idx):
        i = idx[0]
        if i == 0:
            w = 1.0
        else:
            w = (limit - T[i-1]) / (T[i] - T[i-1])
        f['crossed'] = 1
        f['t85'] = t[i-1] + w * (t[i] - t[i-1]) if i > 0 else t[0]
        f['SOC_safe'] = S[i-1] + w * (S[i] - S[i-1]) if i > 0 else S[0]
        f['P_safe'] = P[i-1] + w * (P[i] - P[i-1]) if i > 0 else P[0]
    else:
        f['crossed'] = 0; f['t85'] = np.nan; f['SOC_safe'] = S[-1]; f['P_safe'] = P[-1]
    return f

def build_table(runs):
    rows = []
    for r in runs:
        f = run_features(r['run'], r['T_pre'], r['T_amb'])
        rows.append(dict(tank=r['tank'], T_amb=r['T_amb'], T_pre=r['T_pre'], rate=r['rate'], **f))
    return pd.DataFrame(rows)
