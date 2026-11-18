# h2surrogate

Machine-learning surrogates of 70 MPa hydrogen fills of a 99 L Type 3 / Type 4 tank (trained on H2FillS simulations), a precooling set-point optimiser and a
station-level energy estimate, with a small Streamlit app. Code and data accompanying the manuscript

> A. AitAkka N'alla, M. El Alami, A. Gounni. *Machine-learning surrogates for the safety-limited end state of hydrogen refueling in Type 3 and Type 4 tanks: off-grid and extrapolation validation of six models, and precooling set-points at the 85 °C limit* (submitted).  
> Live app: https://h2surrogate.streamlit.app/  
> Repository DOI: *(to be added after the Zenodo release)*

**Research prototype.** The surrogates reproduce simulations, not measurements. They cover one tank geometry, an initial state of 2 MPa and 25 °C, ambient 20–50 °C,
delivery temperature −40 to +40 °C and ramp rate 10–26 MPa/min. Do not use them to operate a station or a vehicle.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # fits the surrogates in about 15 s, then opens the app
pytest -q tests               # checks that the main paper numbers are reproduced (about 15 s)
```

The app has three tabs: **Predict a fill** (peak temperature with SVR and Gaussian process, SOC and pressure at the 85 °C limit with an MLP),
**Set-point finder** (the delivery temperature and ramp rate of the paper's S2 strategy for a given ambient temperature, with a look-up table to download),
and **Station estimate** (annual precooling electricity, cost and peak duty from 20 years of hourly temperatures of six MENA cities or an uploaded CSV).

## Use as a library

```python
from h2surrogate import SurrogateSet, optimise, station_estimate, load_city

M = SurrogateSet()                                   # fits on data/train_table.csv
M.predict(tank=4, T_amb=35, T_pre=-21, rate=17.5)    # T_peak_svr, T_peak_gp, SOC_safe, P_safe, fill_time
optimise(M, tank=4, T_amb=35, margin=2.0, t_max=240.0, cop_model='carnot')   # {'Tpre': -21.0, 'rate': 17.5, ...}
station_estimate(M, load_city('Riyadh'), tank=4, kg_per_day=100, tariff=0.074)
```

## Layout

| Path | Content |
|---|---|
| `h2surrogate/` | package: models, energy model, set-point optimiser, station estimate |
| `app.py` | Streamlit app |
| `data/` | simulation tables, raw H2FillS files, verification runs, hourly climate (see `data/README.md`) |
| `tests/` | tests against the numbers of the paper |
| `paper_scripts/` | the original analysis scripts (archive, with absolute paths) |

## Method in short

- Inputs: tank type, ambient temperature, delivery temperature, pressure ramp rate. Targets: peak gas temperature, SOC and pressure at the first crossing of 85 °C.
- Models: SVR (RBF, C = 100, ε = 0.1) and a Gaussian process (anisotropic RBF) for the temperature; an MLP (128-64-32) for SOC and pressure. Hyperparameters are fixed, as in the paper.
- Set-point search: lowest precooling electricity subject to predicted T_peak ≤ 85 °C − margin, fill time ≤ limit and T_pre ≤ T_amb, on a 0.5 °C × 0.5 MPa/min grid.
- Electricity: e = c_p (T_amb − T_pre) / (3600 COP), c_p = 14.3 kJ/kg/K; COP constant 1.5 or 0.4 × Carnot with the condenser 10 K above ambient.

## Limitations

Simulated data only; fixed initial tank state; the electricity model is first-order and the economics are electricity only. Six set-points were re-simulated in H2FillS (all within 0.1 °C of the
SVR prediction); the others were not. See the paper for details.

## License and attribution

MIT (see `LICENSE`). The hourly temperature files are derived from ERA5 via Open-Meteo and are released under CC BY 4.0; please attribute Open-Meteo and ERA5.
