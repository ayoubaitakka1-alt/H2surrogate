from fig_common import *
E=pickle.load(open('/home/claude/work/results_econ.pkl','rb'))
C=['Casablanca','Ouarzazate','Cairo','Riyadh','Dubai','Doha']; x=np.arange(6); w=0.36
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0)); a=ax[0]; panel(a,'a')
for j,tk in enumerate((3,4)):
    d=E[E.tank==tk].set_index('city').loc[C]; mid=(d.sav_lo+d.sav_hi)/2; err=(d.sav_hi-d.sav_lo)/2
    a.bar(x+(j-0.5)*w,mid,w,yerr=err,color=TANKC[tk],edgecolor=INK2,lw=0.5,hatch=None if tk==3 else '////',capsize=2,error_kw=dict(lw=0.8,ecolor=INK),label='Type %d'%tk)
a.set_xticks(x); a.set_xticklabels(C,rotation=35,ha='right'); a.set_ylabel('Electricity cost saving (USD/yr)'); a.set_ylim(0,1150); a.legend(frameon=False,loc='upper left',ncol=2)
b=ax[1]; panel(b,'b')
d=E[E.tank==4].set_index('city').loc[C]
s0=[np.mean(v) for v in d.c_kg_S0]; s2=[np.mean(v) for v in d.c_kg_S2]
b.bar(x-w/2,s0,w,color='#c9c8c4',edgecolor=INK2,lw=0.5,label='S0 (−40 °C)'); b.bar(x+w/2,s2,w,color=TANKC[4],edgecolor=INK2,lw=0.5,hatch='////',label='S2, Type 4')
b.set_xticks(x); b.set_xticklabels(C,rotation=35,ha='right'); b.set_ylabel('Precooling electricity cost (US cents/kg)'); b.set_ylim(0,3.8); b.legend(frameon=False,loc='upper left',ncol=2)
fig.tight_layout(); save(fig,'Fig16_economics')
