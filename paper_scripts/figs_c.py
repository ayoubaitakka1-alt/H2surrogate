from fig_common import *
from matplotlib.ticker import FuncFormatter
SP = APP['setpoints']; SENS = APP['sens']
PRE = np.arange(-40, 40.01, 0.5); RATE = np.arange(10, 26.01, 0.5)
a, b = APP['fill_time']['a'], APP['fill_time']['b']
tfill = lambda r: a + b / np.asarray(r, float)
CP = 14.3
def cop(Tpre, Tamb, model):
    if model == 'const': return 1.5
    return 0.4 * (Tpre + 273.15) / (Tamb + 10 + 273.15 - (Tpre + 273.15))
def e_kwh(Tpre, Tamb, model): return CP * np.maximum(Tamb - Tpre, 0) / (3600 * cop(Tpre, Tamb, model))

# ---- fit the SVR once for contours ---------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
sc = StandardScaler().fit(tr[FEATS].values); svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1).fit(sc.transform(tr[FEATS].values), tr.T_peak.values)
def Tgrid(tank, amb):
    PG, RG = np.meshgrid(np.arange(-40, 40.01, 0.5), np.arange(10, 26.01, 0.25), indexing='ij')
    Z = np.column_stack([np.full(PG.size, float(tank == 4)), np.full(PG.size, amb), PG.ravel(), RG.ravel()])
    return PG, RG, svr.predict(sc.transform(Z)).reshape(PG.shape)

# ---------- Fig. 9: compliance maps -----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
for ax, tk, lab in zip(axs, (3, 4), ('(a)', '(b)')):
    PG, RG, T = Tgrid(tk, 35)
    im = ax.contourf(RG, PG, T, levels=np.arange(45, 146, 5), cmap='Blues', vmin=40, vmax=150, alpha=0.95)
    ax.contour(RG, PG, T, levels=[85], colors=['#b83332'], linewidths=1.8)
    ax.contour(RG, PG, T, levels=[83], colors=['#b83332'], linewidths=0.9, linestyles='--')
    for amb, ls in ((20, ':'), (50, '-.')):
        _, _, T2 = Tgrid(tk, amb); ax.contour(RG, PG, T2, levels=[85], colors=[INK], linewidths=0.9, linestyles=ls)
    s = tr[(tr.tank == tk) & (tr.T_amb == 35)]
    ok = s.T_peak <= 85
    ax.scatter(s.rate[ok], s.T_pre[ok], s=14, marker='o', facecolor='white', edgecolor=INK, linewidth=0.7, zorder=4)
    ax.scatter(s.rate[~ok], s.T_pre[~ok], s=14, marker='x', color=INK, linewidth=0.8, zorder=4)
    ax.set_xlabel('Pressure ramp rate (MPa/min)'); panel(ax, lab); ax.set_title(f'Type {tk}, 35 °C ambient', fontsize=8.5); ax.grid(False)
axs[0].set_ylabel('Delivery temperature (°C)')
cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label('SVR peak gas temperature (°C)', fontsize=8); cb.outline.set_visible(False)
h = [Line2D([], [], color='#b83332', lw=1.8, label='85 °C contour'), Line2D([], [], color='#b83332', lw=0.9, ls='--', label='83 °C (2 °C margin)'),
     Line2D([], [], color=INK, lw=0.9, ls=':', label='85 °C at 20 °C ambient'), Line2D([], [], color=INK, lw=0.9, ls='-.', label='85 °C at 50 °C ambient'),
     Line2D([], [], marker='o', ls='', mfc='white', mec=INK, ms=4, label='H2FillS ≤ 85 °C'), Line2D([], [], marker='x', ls='', color=INK, ms=4, label='H2FillS > 85 °C')]
fig.legend(handles=h, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.46, -0.13), fontsize=7.5)
save(fig, 'Fig9_compliance_maps')

# ---------- Fig. 10: selected set-points and time-energy trade-off ------------------------------------------------
PAR = APP['pareto']
fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.0), gridspec_kw=dict(wspace=0.38))
ax = axs[0]; panel(ax, '(a)')
for tk in (3, 4):
    d = SP[(SP.tank == tk) & (SP.cop == 'carnot')]
    ax.plot(d.T_amb, d.S2_Tpre, marker='o', ms=3.5, color=TANKC[tk], label=f'Type {tk}')
ax.axhline(-40, color=INK2, lw=0.9, ls='--'); ax.text(20.5, -37, 'fixed −40 °C (T40)', fontsize=7, color=INK2)
ax.plot([20, 50], [20, 50], color='#9a9994', lw=0.8, ls=':'); ax.text(33, 38, 'ambient', fontsize=7, color=INK2, rotation=0)
ax.set_xlabel('Ambient temperature (°C)'); ax.set_ylabel('Selected delivery temperature (°C)'); ax.legend(frameon=False, loc='center right', fontsize=7.5)
ax = axs[1]; panel(ax, '(b)')
for tk in (3, 4):
    T = PAR[tk]['T']; es, ts = [], []
    for j, r in enumerate(RATE):
        ok = (T[:, j] <= 83) & (PRE <= 35)
        if ok.any(): es.append(e_kwh(PRE[ok].max(), 35, 'carnot')); ts.append(tfill(r))
        else: es.append(np.nan); ts.append(tfill(r))
    ax.plot(ts, es, color=TANKC[tk], lw=1.6)
    ax.plot([tfill(26)], [e_kwh(-40, 35, 'carnot')], 's', color=TANKC[tk], mec=INK, ms=5)
    s2 = SP[(SP.tank == tk) & (SP.cop == 'carnot') & (SP.T_amb == 35)].iloc[0]
    ax.plot([s2.S2_t], [s2.e_S2], '*', color=TANKC[tk], mec=INK, ms=10)
ax.text(330, 0.045, 'Type 3', color=TANKC[3], fontsize=8, ha='center'); ax.text(330, 0.152, 'Type 4', color=TANKC[4], fontsize=8, ha='center')
ax.set_xlabel('Fill time (s)'); ax.set_ylabel('Precooling electricity (kWh/kg)'); ax.set_ylim(-0.01, 0.31)
ax.legend(handles=[Line2D([], [], marker='s', ls='', mfc='#cfcdc8', mec=INK, ms=5, label='T40, 26 MPa/min'), Line2D([], [], marker='*', ls='', mfc='#cfcdc8', mec=INK, ms=10, label='selected (≤ 240 s)')], frameon=False, fontsize=7, loc='upper right'); ax.set_title('35 °C ambient', fontsize=8, color=INK2)
ax = axs[2]; panel(ax, '(c)')
for tk in (3, 4):
    for cm, ls in (('carnot', '-'), ('const', '--')):
        d = SP[(SP.tank == tk) & (SP.cop == cm)]; ax.plot(d.T_amb, d.saving, ls=ls, marker='o', ms=3, color=TANKC[tk])
ax.set_xlabel('Ambient temperature (°C)'); ax.set_ylabel('Electricity saving vs. T40 (%)'); ax.set_ylim(0, 150); ax.set_yticks([0, 20, 40, 60, 80, 100])
h = [Line2D([], [], color=TANKC[3], label='Type 3'), Line2D([], [], color=TANKC[4], label='Type 4'), Line2D([], [], color=INK2, ls='-', label='Carnot-fraction COP'), Line2D([], [], color=INK2, ls='--', label='constant COP 1.5')]
ax.legend(handles=h, frameon=False, fontsize=6.8, loc='upper center', ncol=2, columnspacing=1.0, handlelength=1.6)
save(fig, 'Fig10_setpoints_energy')

# ---------- Fig. 11: sensitivity ---------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, cm, ttl, lab in zip(axs, ('carnot', 'const'), ('Carnot-fraction COP (η = 0.4)', 'Constant COP = 1.5'), ('(a)', '(b)')):
    for tk in (3, 4):
        for mg, ls in ((2.0, '-'), (5.0, '--')):
            d = SENS[(SENS.tank == tk) & (SENS.margin == mg) & (SENS.cop == cm)]
            ax.plot(d.tmax, d.saving_mean, ls=ls, marker='o', ms=3.5, color=TANKC[tk])
    ax.axvline(240, color='#9a9994', lw=0.8, ls=':'); ax.set_xlabel('Maximum allowed fill time (s)'); ax.set_title(ttl, fontsize=8.5); panel(ax, lab)
axs[0].set_ylabel('Mean electricity saving, 20–50 °C ambient (%)'); axs[0].set_ylim(0, 100)
h = [Line2D([], [], color=TANKC[3], label='Type 3'), Line2D([], [], color=TANKC[4], label='Type 4'), Line2D([], [], color=INK2, ls='-', label='2 °C margin'), Line2D([], [], color=INK2, ls='--', label='5 °C margin')]
axs[1].legend(handles=h, frameon=False, fontsize=7.5, loc='center right')
fig.subplots_adjust(wspace=0.08)
save(fig, 'Fig11_sensitivity')

# ---------- Fig. S1: residuals ------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0)); lab = iter(['(a)', '(b)', '(c)', '(d)'])
def resid(tgt, m): return M['C'][(tgt, m)]['pred'] - va[tgt].values
r = resid('T_peak', 'SVR (RBF)')
ax = axs[0, 0]; panel(ax, next(lab)); ax.scatter(va.T_pre, r, s=9, color=COL['SVR (RBF)'], edgecolor=INK2, linewidth=0.3); ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel('Delivery temperature (°C)'); ax.set_ylabel('SVR residual, T_peak (°C)')
ax = axs[0, 1]; panel(ax, next(lab)); ax.scatter(va.T_amb, r, s=9, color=COL['SVR (RBF)'], edgecolor=INK2, linewidth=0.3); ax.axhline(0, color=INK2, lw=0.8)
ax.axvspan(50, 54, color='#d9d8d3', alpha=0.6, lw=0); ax.text(52, -0.45, 'outside\ntraining\nrange', ha='center', va='center', fontsize=6.5, color=INK2); ax.set_xlabel('Ambient temperature (°C)'); ax.set_ylabel('SVR residual, T_peak (°C)')
for k, (m, tgt, un) in enumerate([('MLP', 'SOC_safe', '%'), ('SVR (RBF)', 'SOC_safe', '%')]):
    ax = axs[1, k]; panel(ax, next(lab)); rr = resid(tgt, m); ax.scatter(va.T_pre, rr, s=9, color=COL[m], edgecolor=INK2, linewidth=0.3); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel('Delivery temperature (°C)'); ax.set_ylabel(f'{SHORT[m]} residual, SOC_safe (%)')
fig.subplots_adjust(hspace=0.35, wspace=0.3); save(fig, 'FigS1_residuals')

# ---------- Fig. S2: permutation importance --------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
inputs = ['T_pre', 'tank', 'rate', 'T_amb']; nice = {'T_pre': 'Delivery temp.', 'tank': 'Tank type', 'rate': 'Ramp rate', 'T_amb': 'Ambient temp.'}
for ax, tgt, ttl, unit, lab in zip(axs, ['T_peak', 'SOC_safe', 'P_safe'], ['Peak gas temperature', 'Safety-limited SOC', 'Safety-limited pressure'], ['°C', '%', 'MPa'], ['(a)', '(b)', '(c)']):
    ms = ['SVR (RBF)', 'MLP', 'XGBoost']; w = 0.26
    for k, m in enumerate(ms):
        v = [E['PI'][(tgt, m, i)][0] for i in inputs]; ax.barh(np.arange(4) + (k - 1) * w, v, w * 0.9, color=COL[m], edgecolor=INK2, linewidth=0.4, hatch='///' if m == 'XGBoost' else None, label=SHORT[m], zorder=3)
    ax.set_xscale('symlog', linthresh=0.1); ax.set_yticks(range(4)); ax.set_yticklabels([nice[i] for i in inputs]); ax.invert_yaxis()
    ax.set_xlabel(f'RMSE increase ({unit})'); ax.set_title(ttl, fontsize=8.5); panel(ax, lab); ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
axs[0].legend(frameon=False, fontsize=7.5, loc='lower right', bbox_to_anchor=(1.0, -0.02))
fig.subplots_adjust(wspace=0.12); save(fig, 'FigS2_permutation_importance')

# ---------- Graphical abstract --------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9), gridspec_kw=dict(wspace=0.55))
ax = axs[0]; ms = ['XGBoost', 'SVR (RBF)', 'MLP']; x = np.arange(3); w = 0.36
ax.bar(x - w / 2, [M['A'][('T_peak', m)]['mean']['RMSE'] for m in ms], w, color='#cfcdc8', edgecolor=INK2, linewidth=0.4, label='random CV', zorder=3)
ax.bar(x + w / 2, [M['C'][('T_peak', m)]['m']['RMSE'] for m in ms], w, color=[COL[m] for m in ms], edgecolor=INK2, linewidth=0.4, label='off-grid', zorder=3)
ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ms]); ax.set_ylabel('RMSE, peak gas temp. (°C)'); ax.set_yscale('log'); ax.set_ylim(0.05, 6); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
ax.set_title('Random CV understates the error', fontsize=8.5); ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc='#cfcdc8', ec=INK2, lw=0.4), plt.Rectangle((0, 0), 1, 1, fc='#8a8986', ec=INK2, lw=0.4)], labels=['random CV (grey)', 'off-grid (colour)'], frameon=False, fontsize=7, loc='upper right'); ax.grid(axis='x', visible=False)
ax = axs[1]; ms = ['Random Forest', 'XGBoost', 'LightGBM', 'SVR (RBF)', 'MLP']
v = [E['CL'][(m, 'offgrid')]['false_safe'] for m in ms]
ax.bar(range(5), v, color=[COL[m] for m in ms], edgecolor=INK2, linewidth=0.4, hatch=None, zorder=3)
for i, m in enumerate(ms):
    if m in ('Random Forest', 'XGBoost', 'LightGBM'): ax.patches[i].set_hatch('///')
ax.set_xticks(range(5)); ax.set_xticklabels(['RF', 'XGB', 'LGBM', 'SVR', 'MLP'], fontsize=7.5); ax.set_ylabel('Fills above 85 °C passed\nas compliant (of 131)'); ax.set_ylim(0, 13)
for i, val in enumerate(v): ax.text(i, val + 0.25, str(int(val)), ha='center', va='bottom', fontsize=8, color=INK)
ax.set_title('Off-grid safety errors (223 runs)', fontsize=8.5); ax.grid(axis='x', visible=False)
ax = axs[2]
d3 = SP[(SP.tank == 3) & (SP.cop == 'const')].saving.mean(); d4 = SP[(SP.tank == 4) & (SP.cop == 'const')].saving.mean(); c3 = SP[(SP.tank == 3) & (SP.cop == 'carnot')].saving.mean(); c4 = SP[(SP.tank == 4) & (SP.cop == 'carnot')].saving.mean()
ax.bar([0, 1], [d3, d4], 0.36, color=[TANKC[3], TANKC[4]], edgecolor=INK2, linewidth=0.4, zorder=3, label='constant COP')
ax.bar([0.4, 1.4], [c3, c4], 0.36, color=[TANKC[3], TANKC[4]], edgecolor=INK2, linewidth=0.4, hatch='///', zorder=3, alpha=0.75)
ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(['Type 3', 'Type 4']); ax.set_ylabel('Precooling electricity saved (%)'); ax.set_ylim(0, 125); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title('Estimated saving vs. T40', fontsize=8.5); ax.grid(axis='x', visible=False)
ax.text(0.5, 0.985, 'solid: constant COP\nhatched: Carnot-fraction COP', transform=ax.transAxes, ha='center', va='top', fontsize=6.5, color=INK2, linespacing=1.3)
save(fig, 'Graphical_abstract')
