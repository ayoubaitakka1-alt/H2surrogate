import pickle, json, numpy as np
M=pickle.load(open('/home/claude/work/results_mena.pkl','rb')); R,clip=M['R'],M['clip']
C=['Casablanca','Ouarzazate','Cairo','Riyadh','Dubai','Doha']
g=lambda c,t,cm: R[(R.city==c)&(R.tank==t)&(R.cop==cm)].iloc[0]
t14=[[c,f"{clip.loc[c,'Tmean']:.1f}",f"{clip.loc[c,'Tmax']:.1f}",f"{clip.loc[c,'below']:.0f}"] for c in C]
t15=[];t16=[]
for c in C:
    a3,a4,b3,b4=g(c,3,'carnot'),g(c,4,'carnot'),g(c,3,'const'),g(c,4,'const')
    t15.append([c,f"{a3.e_S0:.3f}",f"{a3.e_S2:.3f} / {a4.e_S2:.3f}",f"{a3.saving:.0f} / {a4.saving:.0f}",f"{b3.e_S0:.3f}",f"{b3.e_S2:.3f} / {b4.e_S2:.3f}",f"{b3.saving:.0f} / {b4.saving:.0f}"])
    t16.append([c,f"{a3.MWh_S0:.1f}",f"{a3.MWh_S2:.2f} / {a4.MWh_S2:.2f}",f"{b3.MWh_S0:.1f}",f"{b3.MWh_S2:.2f} / {b4.MWh_S2:.2f}",f"{a3.q0_peak:.0f}",f"{a3.q2_peak:.1f} / {a4.q2_peak:.1f}",f"{a3.S1_T:.0f} / {a4.S1_T:.0f}",f"{a3.S1_SOC:.0f} / {a4.S1_SOC:.0f}"])
json.dump(dict(t14=t14,t15=t15,t16=t16),open('/home/claude/work/tables_mena.json','w'),ensure_ascii=False)
for t in (t14,t15,t16):
    for r in t: print(r)
