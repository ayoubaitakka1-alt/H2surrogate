import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')

DATA = '/home/claude/work/data/'
FEATS = ['tank4', 'T_amb', 'T_pre', 'rate']
MODELS = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'SVR (RBF)', 'MLP']
SEED = 42

def load_tables():
    tr = pd.read_csv(DATA + 'train_table.csv'); va = pd.read_csv(DATA + 'val_table.csv'); sa = pd.read_csv(DATA + 'setA_table.csv')
    for d in (tr, va, sa):
        d['tank4'] = (d.tank == 4).astype(float)
    return tr, va, sa

def make(name, seed=SEED):
    """Hyper-parameters fixed a priori (IJHE Table 3)."""
    if name == 'Linear Regression': return LinearRegression()
    if name == 'Random Forest': return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1)
    if name == 'XGBoost': return xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, verbosity=0, n_jobs=4)
    if name == 'LightGBM': return lgb.LGBMRegressor(n_estimators=500, max_depth=4, num_leaves=15, learning_rate=0.03, min_child_samples=2, random_state=seed, verbose=-1, n_jobs=4)
    if name == 'SVR (RBF)': return SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    if name == 'MLP': return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', learning_rate_init=0.005, alpha=0.01, max_iter=5000, random_state=seed)
    raise ValueError(name)

def fit_predict(name, Xtr, ytr, Xte, seed=SEED):
    """Scaling is fitted on the training portion only (SVR and MLP); MLP target is standardised."""
    m = make(name, seed)
    if name in ('SVR (RBF)', 'MLP'):
        sx = StandardScaler().fit(Xtr); a, b = sx.transform(Xtr), sx.transform(Xte)
        if name == 'MLP':
            sy = StandardScaler().fit(ytr.reshape(-1, 1))
            m.fit(a, sy.transform(ytr.reshape(-1, 1)).ravel())
            return sy.inverse_transform(m.predict(b).reshape(-1, 1)).ravel()
        m.fit(a, ytr); return m.predict(b)
    m.fit(Xtr, ytr); return m.predict(Xte)

def metrics(y, p):
    e = p - y
    ss_res = np.sum(e**2); ss_tot = np.sum((y - y.mean())**2)
    return dict(R2=1 - ss_res/ss_tot if ss_tot > 0 else np.nan, RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / y)) * 100) if np.all(y != 0) else np.nan, MAXE=float(np.max(np.abs(e))), BIAS=float(np.mean(e)))
