import pickle, json, pandas as pd
M=pickle.load(open('../results_mena.pkl','rb')); R=M['R']
R['dq']=R.q0_peak-R.q2_peak; R['dMWh']=R.MWh_S0-R.MWh_S2
# energy saving per 100 kg/day (MWh/yr), range over sites and COP models
dm={tk:(R[R.tank==tk].dMWh.min(),R[R.tank==tk].dMWh.max()) for tk in (3,4)}
dq={tk:(R[R.tank==tk].dq.min(),R[R.tank==tk].dq.max()) for tk in (3,4)}
print(dm,dq)
sizes=[100,500,1000,2000]; tars=[0.04,0.08,0.12,0.20]
t18=[]
for tk in (4,3):
    for s in sizes:
        row=[f"Type {tk}",f"{s}"]
        for t in tars:
            lo=dm[tk][0]*s/100*1000*t; hi=dm[tk][1]*s/100*1000*t
            row.append(f"{lo:,.0f}–{hi:,.0f}")
        t18.append(row)
for r in t18: print(r)
CRF=0.08*1.08**10/(1.08**10-1); print('CRF',CRF)
ref=70000/13   # USD per kW of refrigeration, uninstalled, single reference design
t19=[]
for tk in (3,4):
    lo,hi=dq[tk]
    for f,label in ((0.5,'half of reference'),(1.0,'reference')):
        u=ref*f
        capex=(lo*u,hi*u); ann=(capex[0]*CRF,capex[1]*CRF)
        e=(dm[tk][0]*1000*0.08, dm[tk][1]*1000*0.08)   # 100 kg/day, 0.08 USD/kWh
        t19.append([f"Type {tk}",f"{lo:.0f}–{hi:.0f}",f"{u:,.0f}",f"{capex[0]/1000:.0f}–{capex[1]/1000:.0f}",f"{ann[0]:,.0f}–{ann[1]:,.0f}",f"{e[0]:.0f}–{e[1]:.0f}"])
for r in t19: print(r)
print('ref USD/kW',ref)
json.dump(dict(t18=t18,t19=t19,ref=ref,crf=CRF,dm=dm,dq=dq),open('../tables_scen.json','w'),ensure_ascii=False)
