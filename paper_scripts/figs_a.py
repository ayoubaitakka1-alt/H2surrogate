from fig_common import *
import loader

# ---------------- Fig. 1: design space and the safety-based targets --------------------------------
runs = loader.load_training(DATA + 'train_type4.xlsx', 4)
ex = next(r for r in runs if r['T_amb'] == 35 and r['T_pre'] == 0 and abs(r['rate'] - 18) < 0.01)['run']
f = loader.run_features(ex, 0, 35)
fig = plt.figure(figsize=(7.4, 5.4)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
ax = fig.add_subplot(gs[0, 0]); panel(ax, '(a)')
g = tr.drop_duplicates(['T_pre', 'rate'])
ax.scatter(g.T_pre, g.rate, s=16, color='#9a9994', label='Training grid (81 per ambient level and tank)', zorder=3)
v = va.drop_duplicates(['T_pre', 'rate'])
ax.scatter(v.T_pre, v.rate, s=34, marker='D', color='#2a78d6', edgecolor=INK, linewidth=0.5, label='Off-grid validation (223 runs)', zorder=4)
ax.scatter(sa.T_pre, sa.rate, s=40, marker='*', color='#eb6834', edgecolor=INK, linewidth=0.5, label='Confirmation set (16 runs)', zorder=5)
ax.set_xlabel('Delivery temperature (°C)'); ax.set_ylabel('Pressure ramp rate (MPa/min)'); ax.set_ylim(8, 40)
ax.legend(frameon=False, loc='upper center', fontsize=7, handlelength=1.2, ncol=1, borderaxespad=0.2)
t = ex.t.values
for k, (col, lab, key, unit, lim) in enumerate([('T', '(b)', 'T_safe', 'Gas temperature (°C)', 85), ('P', '(c)', 'P_safe', 'Tank pressure (MPa)', None), ('SOC', '(d)', 'SOC_safe', 'State of charge (%)', None)]):
    ax = fig.add_subplot(gs[[0, 1, 1][k], [1, 0, 1][k]]); panel(ax, lab)
    ax.plot(t, ex[col].values, color='#2a78d6')
    ax.axvline(f['t85'], color=INK2, lw=0.9, ls=':')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(unit)
    if col == 'T':
        ax.axhline(85, color='#e34948', lw=1.0, ls='--'); ax.text(2, 86.5, '85 °C limit', color='#b83332', fontsize=7.5)
        ax.plot([f['t85']], [85], 'o', color=INK, ms=4); ax.annotate('$t_{85}$ = %.0f s' % f['t85'], (f['t85'], 85), (f['t85'] + 18, 55), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2))
        ax.annotate('$T_{peak}$ = %.1f °C (end of fill)' % f['T_peak'], (t[-1], f['T_peak']), (t[-1] - 4, 44), fontsize=8, va='top', ha='right')
        ax.plot([t[-1]], [f['T_peak']], 'o', color='#2a78d6', ms=4)
    elif col == 'P':
        ax.plot([f['t85']], [f['P_safe']], 'o', color=INK, ms=4); ax.annotate('$P_{safe}$ = %.1f MPa' % f['P_safe'], (f['t85'], f['P_safe']), (f['t85'] + 25, f['P_safe'] - 17), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2))
    else:
        ax.plot([f['t85']], [f['SOC_safe']], 'o', color=INK, ms=4); ax.annotate('$SOC_{safe}$ = %.1f %%' % f['SOC_safe'], (f['t85'], f['SOC_safe']), (f['t85'] + 25, f['SOC_safe'] - 22), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2))
        ax.plot([t[-1]], [ex.SOC.values[-1]], 's', color='#2a78d6', ms=4); ax.annotate('end of fill: %.1f %%' % ex.SOC.values[-1], (t[-1], ex.SOC.values[-1]), (t[-1] - 4, 50), fontsize=7.5, ha='right')
save(fig, 'Fig1_design_and_targets')
print('example', {k: round(v, 2) for k, v in f.items() if k not in ('monotone',)})

# ---------------- Fig. 2: simulated end states --------------------------------------------------------------
fig = plt.figure(figsize=(7.8, 3.2)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.55, 1.15], wspace=0.22)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]; cax = fig.add_subplot(gs[0, 2])
vmin, vmax = 45, 145
for ax, tk, lab in zip(axs[:2], (3, 4), ('(a)', '(b)')):
    s = tr[(tr.tank == tk) & (tr.T_amb == 35)]
    pv = s.pivot_table(index='T_pre', columns='rate', values='T_peak')
    im = ax.imshow(pv.values, origin='lower', aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([int(c) for c in pv.columns], fontsize=6.5); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels([int(c) for c in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]; ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=5.8, color='white' if val > 100 or val < 0 else INK, fontweight='bold' if val > 85 else 'normal')
    ax.set_title(f'Type {tk}', fontsize=8.5); ax.grid(False); panel(ax, lab)
    ax.set_xlabel('Ramp rate (MPa/min)'); ax.set_ylabel('Delivery temperature (°C)' if tk == 3 else '')
    if tk == 4: ax.set_yticklabels([])
    ax.spines[:].set_visible(False)
cb = fig.colorbar(im, cax=cax); cax.set_title('°C', fontsize=8, pad=4); cb.outline.set_visible(False); cax.grid(False)
ax = axs[2]; panel(ax, '(c)')
for tk, ls in ((3, '-'), (4, '--')):
    for pre, c in ((-40, '#2a78d6'), (0, '#eb6834'), (40, '#1baf7a')):
        s = tr[(tr.tank == tk) & (tr.T_pre == pre) & (tr.rate == 18)].sort_values('T_amb')
        ax.plot(s.T_amb, s.T_peak, ls=ls, color=c, marker='o', ms=3)
ax.set_xlabel('Ambient temperature (°C)'); ax.set_ylabel('Final gas temperature (°C)')
h = [Line2D([], [], color='#2a78d6', label='−40 °C'), Line2D([], [], color='#eb6834', label='0 °C'), Line2D([], [], color='#1baf7a', label='+40 °C'),
     Line2D([], [], color=INK2, ls='-', label='Type 3'), Line2D([], [], color=INK2, ls='--', label='Type 4')]
ax.legend(handles=h, frameon=False, fontsize=6.5, loc='center', bbox_to_anchor=(0.5, 0.80), ncol=3, columnspacing=1.0, handlelength=1.6, title='delivery temp. (colour) and tank (line style)', title_fontsize=6.5)
save(fig, 'Fig2_end_states')
