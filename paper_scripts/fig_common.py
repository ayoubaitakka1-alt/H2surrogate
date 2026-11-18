import sys, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, '/home/claude/work/scripts')
from common import *

FIG = '/home/claude/work/figures/'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
import matplotlib.font_manager as fm
avail = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'] if f in avail), 'DejaVu Sans')
mpl.rcParams.update({'font.family': FONT, 'font.size': 9.5, 'axes.labelsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.grid': True, 'grid.color': '#ececea', 'grid.linewidth': 0.5, 'grid.linestyle': ':', 'axes.axisbelow': True, 'axes.spines.top': True, 'axes.spines.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'mathtext.fontset': 'stix',
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'lines.linewidth': 1.6, 'axes.linewidth': 0.8, 'mathtext.default': 'regular'})
# chart order = palette slot order; colour follows the entity everywhere
ORDER = ['SVR (RBF)', 'MLP', 'Random Forest', 'XGBoost', 'LightGBM', 'Linear Regression']
COL = {'SVR (RBF)': '#2a78d6', 'MLP': '#eb6834', 'Random Forest': '#1baf7a', 'XGBoost': '#eda100', 'LightGBM': '#e87ba4', 'Linear Regression': '#8a8986'}
MK = {'SVR (RBF)': 'o', 'MLP': 's', 'Random Forest': '^', 'XGBoost': 'v', 'LightGBM': 'D', 'Linear Regression': 'X'}
SHORT = {'SVR (RBF)': 'SVR', 'MLP': 'MLP', 'Random Forest': 'RF', 'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'Linear Regression': 'Linear'}
TANKC = {3: '#2a78d6', 4: '#eb6834'}

def panel(ax, s):
    ax.text(0.0, 1.02, '(%s)' % s.strip('()'), transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom', ha='left')
def save(fig, name):
    fig.savefig(FIG + name + '.png'); plt.close(fig); print('saved', name)
M = pickle.load(open('/home/claude/work/results_main.pkl', 'rb'))
E = pickle.load(open('/home/claude/work/results_extras.pkl', 'rb'))
APP = pickle.load(open('/home/claude/work/results_app.pkl', 'rb'))
tr, va, sa = load_tables()
def model_legend(ax, loc='best', ncol=1, **kw):
    h = [Line2D([], [], color=COL[m], marker=MK[m], ms=5, lw=1.4, mec=INK2, mew=0.4, label=SHORT[m]) for m in ORDER]
    return ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, **kw)
