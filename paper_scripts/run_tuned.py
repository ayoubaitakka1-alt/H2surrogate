import sys, time, pickle, numpy as np, pandas as pd, itertools
sys.path.insert(0,'/home/claude/work/scripts')
from common import *
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
tr, va, sa = load_tables(); X, Xv = tr[FEATS].values, va[FEATS].values
cv=KFold(5,shuffle=True,random_state=42)
GRID={
 'Random Forest':(RandomForestRegressor(random_state=42,n_jobs=-1),{'n_estimators':[300,600],'max_depth':[8,12,None],'min_samples_leaf':[1,2],'max_features':[1.0,0.75]}),
 'XGBoost':(xgb.XGBRegressor(random_state=42,verbosity=0,n_jobs=4),{'n_estimators':[300,800],'max_depth':[3,6,10],'learning_rate':[0.05,0.1],'min_child_weight':[1,2]}),
 'LightGBM':(lgb.LGBMRegressor(random_state=42,verbose=-1,n_jobs=4),{'n_estimators':[500],'num_leaves':[15,63],'learning_rate':[0.03,0.1],'min_child_samples':[1,5]}),
 'SVR (RBF)':(make_pipeline(StandardScaler(),SVR(kernel='rbf')),{'svr__C':[10,100,1000],'svr__gamma':[0.03,0.1,0.3,1.0],'svr__epsilon':[0.01,0.1]}),
}
OUT={}; t0=time.time()
for tgt in ['T_peak','SOC_safe']:
    y,yv=tr[tgt].values,va[tgt].values
    for name,(est,g) in GRID.items():
        if tgt=='T_peak' and name!='SVR (RBF)': continue
        gs=GridSearchCV(est,g,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=4).fit(X,y)
        best=gs.best_estimator_; pv=best.predict(Xv); m=metrics(yv,pv)
        # edge-level delivery withholding with the tuned hyper-parameters
        te=tr.T_pre.isin([tr.T_pre.min(),tr.T_pre.max()]).values
        from sklearn.base import clone
        pe=clone(best).fit(X[~te],y[~te]).predict(X[te]); edge=metrics(y[te],pe)['RMSE']
        OUT[(tgt,name)]=dict(params=gs.best_params_,cv=-gs.best_score_,off=m,edge=edge,fs=int(((pv<=85)&(va.T_peak.values>85)).sum()) if tgt=='T_peak' else None)
        pickle.dump(OUT,open('/home/claude/work/results_tuned.pkl','wb'))
        print(tgt,name,gs.best_params_,'cv %.3f off %.3f bias %.2f edge %.2f fs %s'%(-gs.best_score_,m['RMSE'],m['BIAS'],edge,OUT[(tgt,name)]['fs']),round(time.time()-t0),flush=True)
pickle.dump(OUT,open('/home/claude/work/results_tuned.pkl','wb'))
print('DONE')
