import sys, pickle, glob, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/work/scripts')
src = open('/home/claude/work/scripts/run_application.py').read().split('AMB = [20, 25')[0]
exec(compile(src, 'run_application_head', 'exec'))
U='/mnt/user-data/uploads/'
FILES={'Casablanca':'*casablanca.csv','Ouarzazate':'*Ouarzazate.csv','Cairo':'*Cairo.csv','Riyadh':'*Riyadh.csv','Dubai':'*Dubai.csv','Doha':'*Doha.csv'}
DOHA = sys.argv[1] if len(sys.argv)>1 else None
LO, HI, MASS = 20.0, 50.0, 3.3
cache = {}
def opt(tank, Ta, cm):
    k = (tank, round(Ta*4)/4, cm)
    if k not in cache: cache[k] = optimise(tank, k[1], 2.0, 240.0, cm)
    return cache[k]
H = {}; 
for c, pat in FILES.items():
    f = DOHA if (c=='Doha' and DOHA) else sorted(glob.glob(U+pat))[-1]
    d = pd.read_csv(f, skiprows=3); d.columns=['t','v']; d['t']=pd.to_datetime(d.t)
    hdr = open(f).read().split('\n')[1].split(',')
    d = d[(d.t.dt.hour>=6)&(d.t.dt.hour<22)].copy()
    d['month']=d.t.dt.month; d['hour']=d.t.dt.hour; d['Traw']=d.v; d['Tuse']=np.clip(d.v,LO,HI); d['Tg']=(d.Tuse*4).round()/4
    H[c]=d; print(c, f.split('/')[-1], 'lat/lon', hdr[0][:6], hdr[1][:6], 'elev', hdr[2], 'hours', len(d))
# profile table for figures
prof=[]; clip={}
for c,d in H.items():
    g=d.groupby(['month','hour']).Traw.mean().reset_index()
    g['T_raw']=g.Traw; g['T_use']=np.clip(g.Traw,LO,HI); g['city']=c; g['days']=g.month.map(dict(zip(range(1,13),[31,28,31,30,31,30,31,31,30,31,30,31])))
    prof.append(g[['city','month','hour','T_raw','T_use','days']])
    clip[c]=dict(below=100*(d.Traw<LO).mean(), above=100*(d.Traw>HI).mean(), Tmean=d.Traw.mean(), Tmax=d.Traw.max(), Tp99=d.Traw.quantile(.99))
P=pd.concat(prof,ignore_index=True); clip=pd.DataFrame(clip).T; print(clip.round(1))
res=[]; mon=[]
for c,d in H.items():
    vc=d.groupby('Tg').size(); Ts=vc.index.values; w=vc.values.astype(float)
    Tmaxg=Ts.max()
    for tank in (3,4):
        for cm in ('carnot','const'):
            S=[opt(tank,T,cm) for T in Ts]
            e0=np.array([float(e_kwh(-40.0,T,cm)) for T in Ts]); e2=np.array([s['e'] for s in S]); t2=np.array([s['t'] for s in S]); tp=np.array([s['Tpre'] for s in S]); rt=np.array([s['rate'] for s in S])
            q0=MASS*CP*(Ts+40)/t_fill(26); q2=MASS*CP*np.maximum(Ts-tp,0)/t2
            r1=float(np.ceil(coef[1]/(240-coef[0]))); Z=np.column_stack([np.full(len(Ts),float(tank==4)),Ts,np.minimum(Ts,40.0),np.full(len(Ts),r1)])
            T1=svrT.predict(Z); S1=mlpS.predict(Z); wa=lambda x: float(np.average(x,weights=w))
            res.append(dict(city=c,tank=tank,cop=cm,e_S0=wa(e0),e_S2=wa(e2),saving=100*(1-wa(e2)/wa(e0)),t_S2=wa(t2),Tpre_S2=wa(tp),rate_S2=wa(rt),q0_peak=float(q0.max()),q2_peak=float(q2.max()),
                            S1_T=wa(T1),S1_SOC=wa(S1),S1_full=100*float(np.average(S1>=99.5,weights=w))))
            for m in range(1,13):
                dm=d[d.month==m]; v=dm.groupby('Tg').size(); tt=v.index.values; ww=v.values.astype(float)
                mon.append(dict(city=c,tank=tank,cop=cm,month=m,e_S0=float(np.average([float(e_kwh(-40.0,T,cm)) for T in tt],weights=ww)),e_S2=float(np.average([opt(tank,T,cm)['e'] for T in tt],weights=ww)),Tpre=float(np.average([opt(tank,T,cm)['Tpre'] for T in tt],weights=ww))))
R=pd.DataFrame(res); R['MWh_S0']=R.e_S0*100*365/1000; R['MWh_S2']=R.e_S2*100*365/1000
MON=pd.DataFrame(mon)
LUT=pd.DataFrame([dict(tank=tk,T_amb=Ta,**{k:v for k,v in opt(tk,float(Ta),'carnot').items() if k in('Tpre','rate','t','T')}) for tk in (3,4) for Ta in range(20,51,2)])
print(R.round(3).to_string())
pickle.dump(dict(R=R,MON=MON,P=P,clip=clip,LUT=LUT,CITY={c:{} for c in H}),open('/home/claude/work/results_mena2.pkl','wb'))
