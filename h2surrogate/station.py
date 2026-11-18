import os
import numpy as np, pandas as pd
from .energy import e_kwh, CP_H2
from .optimise import optimise
from .models import DATA, LIMITS

CITIES = {'Casablanca': 'casablanca', 'Ouarzazate': 'ouarzazate', 'Cairo': 'cairo', 'Riyadh': 'riyadh', 'Dubai': 'dubai', 'Doha': 'doha'}
FILL_KG = 3.3  # hydrogen added to the 99 L tank in one fill


def load_city(name):
    """Hourly 2 m temperature 2005-2024 (ERA5 via Open-Meteo, CC BY 4.0), local time."""
    d = pd.read_csv(os.path.join(DATA, 'climate', f'{CITIES[name]}_2005_2024.csv.gz'), parse_dates=['time'])
    return d.rename(columns={'temperature_2m': 'T'})[['time', 'T']]


def read_temperature_csv(f):
    """Read a CSV with a time column and a temperature column (Open-Meteo files with their 3 header lines are accepted)."""
    text = f.read().decode('utf-8') if hasattr(f, 'read') else open(f).read()
    lines = text.splitlines(); k = next(i for i, l in enumerate(lines) if l.lower().startswith('time'))
    from io import StringIO
    d = pd.read_csv(StringIO('\n'.join(lines[k:])))
    d = d.iloc[:, :2]; d.columns = ['time', 'T']; d['time'] = pd.to_datetime(d['time'])
    return d


def station_estimate(models, hourly, tank, kg_per_day=100.0, hours=(6, 22), tariff=0.10, cop_model='carnot', margin=2.0, t_max=240.0):
    """Annual precooling electricity, cost and peak duty of a station for S0 (-40 degC, 26 MPa/min) and S2 (surrogate-selected).

    hourly : DataFrame with columns time, T (degC); every hour inside `hours` is one equally weighted operating hour.
    Ambient temperatures are limited to the training range 20-50 degC.
    """
    d = hourly[(hourly.time.dt.hour >= hours[0]) & (hourly.time.dt.hour < hours[1])].copy()
    lo, hi = LIMITS['T_amb']
    below = float((d['T'] < lo).mean() * 100); above = float((d['T'] > hi).mean() * 100)
    d['Tg'] = (np.clip(d['T'], lo, hi) * 4).round() / 4
    d['month'] = d.time.dt.month
    vc = d.groupby('Tg').size(); Ts = vc.index.values; w = vc.values.astype(float)
    S = [optimise(models, tank, T, margin, t_max, cop_model) for T in Ts]
    if any(s is None for s in S):
        raise ValueError('No feasible set-point at some ambient temperatures; relax the margin or raise the fill-time limit.')
    e0 = e_kwh(-40.0, Ts, cop_model); e2 = np.array([s['e'] for s in S]); tp = np.array([s['Tpre'] for s in S])
    rt = np.array([s['rate'] for s in S]); t2 = np.array([s['t'] for s in S])
    wa = lambda x: float(np.average(x, weights=w))
    q0 = FILL_KG * CP_H2 * (Ts + 40) / models.fill_time(26.0); q2 = FILL_KG * CP_H2 * np.maximum(Ts - tp, 0) / t2
    ann = kg_per_day * 365.0
    res = dict(kWh_per_kg_S0=wa(e0), kWh_per_kg_S2=wa(e2), MWh_yr_S0=wa(e0) * ann / 1000, MWh_yr_S2=wa(e2) * ann / 1000,
               cost_S0=wa(e0) * ann * tariff, cost_S2=wa(e2) * ann * tariff, saving_pct=100 * (1 - wa(e2) / wa(e0)),
               mean_Tpre=wa(tp), mean_rate=wa(rt), mean_fill_time=wa(t2), peak_duty_S0_kW=float(q0.max()), peak_duty_S2_kW=float(q2.max()),
               hours_below_20_pct=below, hours_above_50_pct=above, n_hours=int(len(d)))
    res['cost_saving'] = res['cost_S0'] - res['cost_S2']
    mon = []
    for m in range(1, 13):
        vm = d[d.month == m].groupby('Tg').size()
        if len(vm) == 0: continue
        ix = [list(Ts).index(t) for t in vm.index]; wm = vm.values.astype(float)
        mon.append(dict(month=m, S0=float(np.average(e0[ix], weights=wm)), S2=float(np.average(e2[ix], weights=wm))))
    res['monthly'] = pd.DataFrame(mon)
    return res
