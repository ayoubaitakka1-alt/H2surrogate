import pickle, json, numpy as np, pandas as pd
M=pickle.load(open('../results_mena.pkl','rb')); R=M['R']
TAR={'Casablanca':0.118,'Ouarzazate':0.118,'Cairo':0.041,'Riyadh':0.074,'Dubai':0.110,'Doha':0.036}
C=list(TAR); rows=[]
for c in C:
    for tk in (3,4):
        g={cm:R[(R.city==c)&(R.tank==tk)&(R.cop==cm)].iloc[0] for cm in('carnot','const')}
        d={cm:(g[cm].MWh_S0-g[cm].MWh_S2)*1000*TAR[c] for cm in g}
        rows.append(dict(city=c,tank=tk,tariff=TAR[c],sav_lo=min(d.values()),sav_hi=max(d.values()),sav_carnot=d['carnot'],sav_const=d['const'],
          S0_cost=(g['carnot'].MWh_S0*1000*TAR[c], g['const'].MWh_S0*1000*TAR[c]),S2_cost=(g['carnot'].MWh_S2*1000*TAR[c], g['const'].MWh_S2*1000*TAR[c]),
          c_kg_S0=(g['carnot'].e_S0*TAR[c]*100,g['const'].e_S0*TAR[c]*100),c_kg_S2=(g['carnot'].e_S2*TAR[c]*100,g['const'].e_S2*TAR[c]*100)))
E=pd.DataFrame(rows); print(E.round(1).to_string())
t=[]
for c in C:
    a=E[(E.city==c)&(E.tank==3)].iloc[0]; b=E[(E.city==c)&(E.tank==4)].iloc[0]
    lo=lambda x:min(x); hi=lambda x:max(x)
    t.append([c,f"{TAR[c]:.3f}",f"{lo(a.S0_cost):.0f}–{hi(a.S0_cost):.0f}",f"{a.sav_lo:.0f}–{a.sav_hi:.0f}",f"{b.sav_lo:.0f}–{b.sav_hi:.0f}",f"{lo(a.c_kg_S0):.1f}–{hi(a.c_kg_S0):.1f}",f"{lo(b.c_kg_S2):.1f}–{hi(b.c_kg_S2):.1f}"])
for r in t: print(r)
json.dump(dict(t17=t),open('../tables_econ.json','w'),ensure_ascii=False)
pickle.dump(E,open('../results_econ.pkl','wb'))
