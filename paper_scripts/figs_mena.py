from fig_common import *
MN = pickle.load(open('/home/claude/work/results_mena.pkl','rb')); R, MON, P, LUT = MN['R'], MN['MON'], MN['P'], MN['LUT']
CITIES = ['Casablanca','Ouarzazate','Cairo','Riyadh','Dubai','Doha']
CC = {'Casablanca':'#2a78d6','Ouarzazate':'#eb6834','Cairo':'#1baf7a','Riyadh':'#eda100','Dubai':'#e87ba4','Doha':'#8a8986'}
CM = {'Casablanca':'o','Ouarzazate':'s','Cairo':'^','Riyadh':'v','Dubai':'D','Doha':'X'}
# ---- Fig A: climate and annual electricity
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.05, 1]))
a = ax[0]; panel(a, 'a')
for c in CITIES:
    g = P[P.city == c].groupby('month').T_raw.mean()
    a.plot(g.index, g.values, color=CC[c], marker=CM[c], ms=4, mew=0.4, mec=INK2, label=c)
a.axhline(20, color=INK2, lw=0.8, ls='--'); a.text(7, 19.3, 'surrogate lower limit (20 °C)', fontsize=7, color=INK2, ha='center', va='top')
a.set_xticks(range(1, 13)); a.set_xticklabels(list('JFMAMJJASOND')); a.set_xlabel('Month'); a.set_ylabel('Mean ambient temperature, 06:00–22:00 (°C)')
a.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5, columnspacing=1.0, handletextpad=0.4)
b = ax[1]; panel(b, 'b')
x = np.arange(len(CITIES)); w = 0.19
for i, (tk, col, hatch) in enumerate([(3, TANKC[3], None), (4, TANKC[4], None)]):
    s0 = [R[(R.city == c) & (R.tank == tk) & (R.cop == 'carnot')].MWh_S0.iloc[0] for c in CITIES]
    s2 = [R[(R.city == c) & (R.tank == tk) & (R.cop == 'carnot')].MWh_S2.iloc[0] for c in CITIES]
    s2c = [R[(R.city == c) & (R.tank == tk) & (R.cop == 'const')].MWh_S2.iloc[0] for c in CITIES]
    off = (i - 0.5) * 2 * w
if True:
    s0 = [R[(R.city == c) & (R.tank == 3) & (R.cop == 'carnot')].MWh_S0.iloc[0] for c in CITIES]
    b.bar(x - 1.5 * w, s0, w, color='#c9c8c4', edgecolor=INK2, lw=0.5, label='S0 fixed −40 °C')
    for j, tk in enumerate((3, 4)):
        s2 = [R[(R.city == c) & (R.tank == tk) & (R.cop == 'carnot')].MWh_S2.iloc[0] for c in CITIES]
        b.bar(x + (j - 0.5) * w + 0.0 * w, s2, w, color=TANKC[tk], edgecolor=INK2, lw=0.5, label='S2, Type %d' % tk, hatch=None if tk == 3 else '////')
        s2c = [R[(R.city == c) & (R.tank == tk) & (R.cop == 'const')].MWh_S2.iloc[0] for c in CITIES]
        b.plot(x + (j - 0.5) * w, s2c, ls='none', marker='_', color=INK, ms=7, mew=1.4, label='constant COP' if j == 0 else None)
b.set_xticks(x); b.set_xticklabels(CITIES, rotation=35, ha='right'); b.set_ylabel('Precooling electricity (MWh/yr)')
b.legend(frameon=False, loc='upper left', fontsize=7.2, ncol=2, columnspacing=1.0)
b.set_ylim(0, 12.5)
fig.tight_layout(); save(fig, 'Fig14_mena_climate_energy')
# ---- Fig B: dispenser lookup and monthly energy
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
a = ax[0]; panel(a, 'a')
for tk in (3, 4):
    d = LUT[LUT.tank == tk]; a.plot(d.T_amb, d.Tpre, color=TANKC[tk], marker='o' if tk == 3 else 's', ms=3.5, mec=INK2, mew=0.4)
    a.text(50.5, d.Tpre.iloc[-1], 'Type %d' % tk, color=INK, fontsize=7.5, va='center')
a.axhline(-40, color=INK2, ls='--', lw=0.8); a.text(20.5, -39, 'S0: −40 °C', fontsize=7, color=INK2, va='bottom')
a.set_xlim(19, 58); a.set_xlabel('Ambient temperature (°C)'); a.set_ylabel('Selected delivery temperature (°C)')
a = ax[1]; panel(a, 'b')
for tk in (3, 4):
    d = LUT[LUT.tank == tk]; a.plot(d.T_amb, d.rate, color=TANKC[tk], marker='o' if tk == 3 else 's', ms=3.5, mec=INK2, mew=0.4)
a.axhline(26, color=INK2, ls='--', lw=0.8); a.text(20.5, 25.4, 'S0: 26', fontsize=7, color=INK2, va='top'); a.set_ylim(15, 27.5)
a.set_xlabel('Ambient temperature (°C)'); a.set_ylabel('Selected ramp rate (MPa/min)')
a = ax[2]; panel(a, 'c')
for c in ('Casablanca', 'Riyadh'):
    for tk, ls in ((3, '-'), (4, '--')):
        d = MON[(MON.city == c) & (MON.tank == tk) & (MON.cop == 'carnot')]
        a.plot(d.month, d.e_S2, color=CC[c], ls=ls, marker=CM[c], ms=3.5, mec=INK2, mew=0.4)
    d = MON[(MON.city == c) & (MON.tank == 3) & (MON.cop == 'carnot')]
    a.plot(d.month, d.e_S0, color=CC[c], ls=':', lw=1.2)
a.set_xticks([1, 4, 7, 10]); a.set_xlabel('Month'); a.set_ylabel('Electricity (kWh/kg)')
h = [Line2D([], [], color=CC['Casablanca'], marker='o', ms=4, mec=INK2, mew=0.4, label='Casablanca'), Line2D([], [], color=CC['Riyadh'], marker='v', ms=4, mec=INK2, mew=0.4, label='Riyadh'),
     Line2D([], [], color=INK2, ls='-', label='S2 Type 3'), Line2D([], [], color=INK2, ls='--', label='S2 Type 4'), Line2D([], [], color=INK2, ls=':', label='S0')]
a.legend(handles=h, frameon=False, fontsize=6.8, loc='upper left', ncol=2, columnspacing=0.8)
a.set_ylim(0, 0.45)
fig.tight_layout(); save(fig, 'Fig15_mena_dispenser_table')
