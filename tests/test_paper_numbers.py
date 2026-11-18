"""Checks that the package reproduces numbers reported in the manuscript."""
import numpy as np, pandas as pd, pytest
from h2surrogate import SurrogateSet, optimise, station_estimate, load_city, FEATS


@pytest.fixture(scope='module')
def M():
    return SurrogateSet()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def test_offgrid_temperature(M):
    va = M.val; X = va[FEATS].values
    assert rmse(va.T_peak, M.svr_T.predict(X)) == pytest.approx(0.104, abs=0.015)     # Table 5: SVR
    assert rmse(va.T_peak, M.gp_T.predict(X)) == pytest.approx(0.052, abs=0.02)       # Sec. 3.11: Gaussian process


def test_offgrid_soc(M):
    va = M.val
    assert rmse(va.SOC_safe, M.mlp_S.predict(va[FEATS].values)) == pytest.approx(0.45, abs=0.15)  # MLP


def test_no_false_safe(M):
    va = M.val; p = M.svr_T.predict(va[FEATS].values)
    assert int(((p <= 85) & (va.T_peak > 85)).sum()) == 0


def test_setpoints(M):
    s = optimise(M, 4, 35, 2.0, 240.0, 'carnot'); assert (s['Tpre'], s['rate']) == (-21.0, 17.5)    # SP11
    s = optimise(M, 3, 20, 2.0, 240.0, 'carnot'); assert (s['Tpre'], s['rate']) == (16.0, 17.5)     # SP01


def test_verification_runs(M):
    v = pd.read_csv('data/verification_results.csv')
    for _, r in v.iterrows():
        p = M.predict(int(r.tank), r.amb, r.Tpre, r.rate)['T_peak_svr']
        assert abs(p - r.Tsim) < 0.2, r.id          # H2FillS end temperature vs surrogate


def test_station_riyadh(M):
    res = station_estimate(M, load_city('Riyadh'), 4, 100.0, (6, 22), 0.074, 'carnot')
    assert res['MWh_yr_S2'] == pytest.approx(4.52, abs=0.05)
    assert res['saving_pct'] == pytest.approx(48.2, abs=0.5)
