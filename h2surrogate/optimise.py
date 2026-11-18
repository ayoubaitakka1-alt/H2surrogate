import numpy as np
from .energy import e_kwh

PRE = np.arange(-40, 40.01, 0.5)
RATE = np.arange(10, 26.01, 0.5)
PG, RG = np.meshgrid(PRE, RATE, indexing='ij')


def optimise(models, tank, T_amb, margin=2.0, t_max=240.0, cop_model='carnot'):
    """Set-point search of the paper: the lowest precooling electricity subject to T_peak <= 85 - margin (SVR surrogate),
    fill time <= t_max and T_pre <= T_amb; ties (within 1 %) are broken by the shortest fill.
    Returns dict(Tpre, rate, t, T, e) or None when no set-point is feasible."""
    Z = np.column_stack([np.full(PG.size, float(tank == 4)), np.full(PG.size, T_amb), PG.ravel(), RG.ravel()])
    T = models.svr_T.predict(Z).reshape(PG.shape)
    ok = (T <= 85 - margin) & (models.fill_time(RG) <= t_max) & (PG <= T_amb)
    if not ok.any():
        return None
    E = np.where(ok, e_kwh(PG, T_amb, cop_model), np.inf)
    cand = ok & (E <= E.min() * 1.01 + 1e-12)
    tf = np.where(cand, models.fill_time(RG), np.inf)
    i = np.unravel_index(np.argmin(tf), tf.shape)
    return dict(Tpre=float(PG[i]), rate=float(RG[i]), t=float(models.fill_time(RG[i])), T=float(T[i]),
                e=float(e_kwh(PG[i], T_amb, cop_model)))
