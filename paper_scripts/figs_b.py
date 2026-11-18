from fig_common import *
from matplotlib.ticker import FuncFormatter, ScalarFormatter

def rms3(vals): return float(np.sqrt(np.mean(np.square(vals))))
def getprot(tgt, m):
    A = M['A'][(tgt, m)]['mean']['RMSE']
    L = rms3([M['B'][(tgt, m, f)]['m']['RMSE'] for f in ['T_amb', 'T_pre', 'rate']])
    C = M['C'][(tgt, m)]['m']['RMSE']
    S = M['S'][(tgt, m)]['m']['RMSE']
    D = rms3([M['D'][(tgt, m, f)]['m']['RMSE'] for f in ['T_amb', 'T_pre', 'rate']])
    return [A, L, C, S, D]
PROT = ['Random\n5-fold CV', 'Leave-one-\nlevel-out', 'Off-grid\nvalidation\n(223 runs)', 'Confirm.\nset\n(16 runs)', 'Edge\nlevels\nwithheld']

# ---------- Fig. 3: RMSE under the protocols ------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.5), sharex=True)
for ax, (tgt, ttl, unit, lab) in zip(axs, [('T_peak', 'Peak gas temperature', '°C', '(a)'), ('SOC_safe', 'Safety-limited state of charge', '% SOC', '(b)')]):
    w = 0.13; x = np.arange(len(PROT))
    for k, m in enumerate(ORDER):
        v = getprot(tgt, m)
        ax.bar(x + (k - 2.5) * w, v, w * 0.92, color=COL[m], edgecolor=INK2, linewidth=0.4, hatch='///' if m in ('Random Forest', 'XGBoost', 'LightGBM') else None, label=SHORT[m], zorder=3)
    ax.set_yscale('log'); ax.set_ylim(0.03, 30); ax.set_xticks(x); ax.set_xticklabels(PROT, fontsize=6.8)
    ax.set_ylabel(f'RMSE ({unit})'); ax.set_title(ttl, fontsize=8.5); panel(ax, lab); ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
h = [plt.Rectangle((0, 0), 1, 1, fc=COL[m], ec=INK2, lw=0.4, hatch='///' if m in ('Random Forest', 'XGBoost', 'LightGBM') else None) for m in ORDER]
fig.legend(h, [SHORT[m] for m in ORDER], ncol=6, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.07), fontsize=8)
fig.subplots_adjust(wspace=0.28)
save(fig, 'Fig3_rmse_protocols')

# ---------- Fig. 4: off-grid parity, peak gas temperature --------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2))
yv = va.T_peak.values
for ax, m, lab in zip(axs.ravel(), ORDER, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    p = M['C'][('T_peak', m)]['pred']; mm = M['C'][('T_peak', m)]['m']
    lo, hi = 45, 145
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8); ax.fill_between([lo, hi], [lo - 2, hi - 2], [lo + 2, hi + 2], color='#d9d8d3', alpha=0.6, lw=0, zorder=1)
    for tk, mk in ((3, 'o'), (4, '^')):
        s = va.tank.values == tk
        ax.scatter(yv[s], p[s], s=9, marker=mk, color=COL[m], edgecolor=INK2, linewidth=0.3, alpha=0.9, zorder=3)
    ax.axvline(85, color='#e34948', lw=0.7, ls='--'); ax.axhline(85, color='#e34948', lw=0.7, ls='--')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); panel(ax, lab)
    ax.set_title(f'{SHORT[m]}', fontsize=8.5, pad=3)
    ax.text(0.04, 0.95, f"RMSE {mm['RMSE']:.2f} °C\nbias {mm['BIAS']:+.2f} °C\nmax |e| {mm['MAXE']:.1f} °C", transform=ax.transAxes, va='top', fontsize=7)
    if ax in axs[1]: ax.set_xlabel('H2FillS (°C)')
    if ax in axs[:, 0]: ax.set_ylabel('Predicted (°C)')
h = [Line2D([], [], marker='o', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='Type 3'), Line2D([], [], marker='^', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='Type 4'),
     Line2D([], [], color='#d9d8d3', lw=6, label='±2 °C'), Line2D([], [], color='#e34948', ls='--', lw=0.8, label='85 °C limit')]
fig.legend(handles=h, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.03), fontsize=8)
fig.subplots_adjust(hspace=0.25, wspace=0.12)
save(fig, 'Fig4_parity_temperature')

# ---------- Fig. 5: off-grid parity, safety-limited SOC and pressure ------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2))
labs = iter(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'])
for r, (tgt, nm, unit, lo, hi) in enumerate([('SOC_safe', 'Safety-limited SOC', '%', 0, 100), ('P_safe', 'Safety-limited pressure', 'MPa', 0, 75)]):
    for c, m in enumerate(['MLP', 'SVR (RBF)', 'XGBoost']):
        ax = axs[r, c]; p = M['C'][(tgt, m)]['pred']; mm = M['C'][(tgt, m)]['m']; y = va[tgt].values
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        for tk, mk in ((3, 'o'), (4, '^')):
            s = va.tank.values == tk
            ax.scatter(y[s], p[s], s=9, marker=mk, color=COL[m], edgecolor=INK2, linewidth=0.3, alpha=0.9, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); panel(ax, next(labs)); ax.set_title(f'{SHORT[m]} – {nm}', fontsize=7.5, pad=3, loc='right')
        ax.text(0.04, 0.95, f"RMSE {mm['RMSE']:.2f} {unit}\nbias {mm['BIAS']:+.2f} {unit}\nmax |e| {mm['MAXE']:.1f} {unit}", transform=ax.transAxes, va='top', fontsize=7)
        if r == 1: ax.set_xlabel(f'H2FillS ({unit})')
        if c == 0: ax.set_ylabel(f'Predicted ({unit})')
h = [Line2D([], [], marker='o', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='Type 3'), Line2D([], [], marker='^', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='Type 4')]
fig.legend(handles=h, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.03), fontsize=8)
fig.subplots_adjust(hspace=0.28, wspace=0.12)
save(fig, 'Fig5_parity_safety_targets')

# ---------- Fig. 6: why random CV misleads - curves along the precooling axis ------------------------------------
fine = np.arange(-40, 40.1, 1.0)
Z = np.column_stack([np.ones_like(fine), np.full_like(fine, 37.0), fine, np.full_like(fine, 21.0)])
vsel = va[(va.tank == 4) & (va.T_amb == 37) & (va.rate == 21)]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
for ax, tgt, lab, unit in zip(axs, ['T_peak', 'SOC_safe'], ['(a)', '(b)'], ['Peak gas temperature (°C)', 'Safety-limited SOC (%)']):
    y = tr[tgt].values
    for m in ['Random Forest', 'XGBoost', 'MLP', 'SVR (RBF)']:
        p = fit_predict(m, X_ := tr[FEATS].values, y, Z)
        ax.plot(fine, p, color=COL[m], lw=1.4, ls='-', label=SHORT[m], zorder=3 if m in ('SVR (RBF)', 'MLP') else 2, drawstyle='steps-mid' if False else 'default')
    ax.scatter(vsel.T_pre, vsel[tgt], s=34, marker='D', color='white', edgecolor=INK, linewidth=1.1, zorder=5, label='H2FillS (off-grid)')
    ax.set_xlabel('Delivery temperature (°C)'); ax.set_ylabel(unit); panel(ax, lab)
    if tgt == 'T_peak': ax.axhline(85, color='#e34948', lw=0.8, ls='--'); ax.text(-39, 87, '85 °C', color='#b83332', fontsize=7.5)
axs[0].legend(frameon=False, loc='upper left', fontsize=7.5)
fig.suptitle('Type 4 tank, 37 °C ambient, 21 MPa/min (all three off the training grid)', fontsize=8, color=INK2, y=1.0)
fig.subplots_adjust(wspace=0.28)
save(fig, 'Fig6_precooling_axis')

# ---------- Fig. 7: edge extrapolation (precooling extremes withheld) --------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2)); labs = iter(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'])
for r, (tgt, nm, unit, lo, hi) in enumerate([('T_peak', '$T_{peak}$', '°C', 45, 150), ('SOC_safe', '$SOC_{safe}$', '%', 0, 100)]):
    for c, m in enumerate(['XGBoost', 'SVR (RBF)', 'MLP']):
        ax = axs[r, c]; d = M['D'][(tgt, m, 'T_pre')]; y = tr[tgt].values[d['idx']]; p = d['pred']; mm = d['m']
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        pre = tr.T_pre.values[d['idx']]
        for pv, mk in ((-40, 'v'), (40, '^')):
            s = pre == pv; ax.scatter(y[s], p[s], s=9, marker=mk, color=COL[m], edgecolor=INK2, linewidth=0.3, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); panel(ax, next(labs)); ax.set_title(f'{SHORT[m]} – {nm}', fontsize=8.5, pad=3, loc='right')
        ax.text(0.04, 0.95, f"RMSE {mm['RMSE']:.2f} {unit}\nbias {mm['BIAS']:+.2f} {unit}", transform=ax.transAxes, va='top', fontsize=7)
        if r == 1: ax.set_xlabel(f'H2FillS ({unit})')
        if c == 0: ax.set_ylabel(f'Predicted ({unit})')
h = [Line2D([], [], marker='v', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='−40 °C runs (withheld)'), Line2D([], [], marker='^', ls='', color='#9a9994', mec=INK2, mew=0.3, ms=4, label='+40 °C runs (withheld)')]
fig.legend(handles=h, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.03), fontsize=8)
fig.subplots_adjust(hspace=0.28, wspace=0.12)
save(fig, 'Fig7_edge_extrapolation')

# ---------- Fig. 8: learning curves --------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
for ax, tgt, lab, unit in zip(axs, ['T_peak', 'SOC_safe'], ['(a)', '(b)'], ['°C', '% SOC']):
    ns = [50, 100, 200, 400, 800, 1134]
    for m in ORDER:
        v = [E['LC'][(tgt, m, n)][0] for n in ns]; lo = [E['LC'][(tgt, m, n)][1] for n in ns]; hi = [E['LC'][(tgt, m, n)][2] for n in ns]
        ax.plot(ns, v, color=COL[m], marker=MK[m], ms=4, mec=INK2, mew=0.3, lw=1.3)
        ax.fill_between(ns, lo, hi, color=COL[m], alpha=0.12, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(ns); ax.get_xaxis().set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ''))
    ax.tick_params(axis='x', labelsize=7); ax.set_xlabel('Training simulations'); ax.set_ylabel(f'Off-grid RMSE ({unit})'); panel(ax, lab)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
model_legend(axs[1], loc='lower left', ncol=2, fontsize=7)
fig.subplots_adjust(wspace=0.28)
save(fig, 'Fig8_learning_curves')
