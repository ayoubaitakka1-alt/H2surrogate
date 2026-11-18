import os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as CK

DATA = os.environ.get('H2SURROGATE_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
FEATS = ['tank4', 'T_amb', 'T_pre', 'rate']
# range of the training grid (inputs outside it are extrapolation)
LIMITS = dict(T_amb=(20.0, 50.0), T_pre=(-40.0, 40.0), rate=(10.0, 26.0))
SEED = 42


def load_tables(data_dir=DATA):
    tr = pd.read_csv(os.path.join(data_dir, 'train_table.csv'))
    va = pd.read_csv(os.path.join(data_dir, 'val_table.csv'))
    co = pd.read_csv(os.path.join(data_dir, 'confirmation_table.csv'))
    for d in (tr, va, co):
        d['tank4'] = (d.tank == 4).astype(float)
    return tr, va, co


class _Scaled:
    """Model + input scaling (+ optional target scaling for the MLP), as used in the manuscript."""
    def __init__(self, model, X, y, scale_y=False):
        self.sx = StandardScaler().fit(X); self.scale_y = scale_y; self.m = model
        if scale_y:
            self.sy = StandardScaler().fit(y.reshape(-1, 1)); self.m.fit(self.sx.transform(X), self.sy.transform(y.reshape(-1, 1)).ravel())
        else:
            self.m.fit(self.sx.transform(X), y)

    def predict(self, Z, return_std=False):
        Zs = self.sx.transform(np.asarray(Z, float))
        if return_std:
            p, s = self.m.predict(Zs, return_std=True)
            return p, s
        p = self.m.predict(Zs)
        return self.sy.inverse_transform(p.reshape(-1, 1)).ravel() if self.scale_y else p


class SurrogateSet:
    """The surrogates of the paper, fitted on all 1134 training simulations.

    svr_T : support-vector regression of the peak gas temperature (used for the set-point search)
    gp_T  : Gaussian process for the peak gas temperature (gives a standard deviation)
    mlp_S : multi-layer perceptron for the state of charge at the 85 degC limit
    mlp_P : multi-layer perceptron for the pressure at the 85 degC limit
    """
    def __init__(self, data_dir=DATA, fit_gp=True):
        self.train, self.val, self.conf = load_tables(data_dir)
        tr = self.train; X = tr[FEATS].values
        # fill time is a deterministic function of the ramp rate: t = a + b / rate
        A = np.vstack([np.ones(len(tr)), 1 / tr.rate.values]).T
        (self.t_a, self.t_b), *_ = np.linalg.lstsq(A, tr.t_end.values, rcond=None)
        self.svr_T = _Scaled(SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1), X, tr.T_peak.values)
        self.mlp_S = _Scaled(self._mlp(), X, tr.SOC_safe.values, scale_y=True)
        self.mlp_P = _Scaled(self._mlp(), X, tr.P_safe.values, scale_y=True)
        self.gp_T = None
        if fit_gp:
            k = CK(1.0, (1e-2, 1e3)) * RBF([1.0] * 4, (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-8, 1e-1))
            self.gp_T = _Scaled(GPR(k, normalize_y=True, n_restarts_optimizer=0, random_state=0), X, tr.T_peak.values)

    @staticmethod
    def _mlp():
        return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', learning_rate_init=0.005,
                            alpha=0.01, max_iter=5000, random_state=SEED)

    def fill_time(self, rate):
        return self.t_a + self.t_b / np.asarray(rate, float)

    @staticmethod
    def features(tank, T_amb, T_pre, rate):
        return np.column_stack([np.asarray(tank, float) == 4, T_amb, T_pre, rate]).astype(float)

    def out_of_range(self, T_amb, T_pre, rate):
        """List of inputs that lie outside the range of the training simulations."""
        msg = []
        for name, v in (('T_amb', T_amb), ('T_pre', T_pre), ('rate', rate)):
            lo, hi = LIMITS[name]
            if v < lo or v > hi:
                msg.append(f'{name} = {v:g} is outside the training range {lo:g} to {hi:g}')
        return msg

    def predict(self, tank, T_amb, T_pre, rate, margin=2.0):
        """Predict one fill. tank is 3 or 4. Returns a dict (peak temperature, SOC and pressure at 85 degC, fill time)."""
        Z = self.features([tank], [T_amb], [T_pre], [rate])
        out = dict(T_peak_svr=float(self.svr_T.predict(Z)[0]), SOC_safe=float(self.mlp_S.predict(Z)[0]),
                   P_safe=float(self.mlp_P.predict(Z)[0]), fill_time=float(self.fill_time(rate)),
                   warnings=self.out_of_range(T_amb, T_pre, rate))
        if self.gp_T is not None:
            m, s = self.gp_T.predict(Z, return_std=True); out['T_peak_gp'] = float(m[0]); out['T_peak_gp_std'] = float(s[0])
        out['compliant'] = out['T_peak_svr'] <= 85.0 - margin
        out['reaches_limit'] = out['T_peak_svr'] > 85.0
        return out
