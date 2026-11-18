"""Streamlit app for the hydrogen-fill surrogates. Run:  streamlit run app.py"""
import numpy as np, pandas as pd, altair as alt, streamlit as st
import h2surrogate
from h2surrogate import SurrogateSet, optimise, station_estimate, load_city, CITIES, e_kwh
from h2surrogate.station import read_temperature_csv

REPO = 'https://github.com/ayoubaitakka1-alt/H2surrogate'
BLUE, ORANGE, GREY = '#1f5fa8', '#d9622b', '#6b7683'
st.set_page_config(page_title='H2 Fill Surrogates', page_icon=None, layout='wide', initial_sidebar_state='expanded')
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1 {font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.1rem;}
div[data-testid="stMetric"] {background: #f3f5f8; border: 1px solid #e1e6ec; border-radius: 8px; padding: 0.7rem 0.9rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #4a5562;}
.subtitle {color: #4a5562; font-size: 1.02rem; margin-bottom: 0.6rem;}
.notice {color: #6b7683; font-size: 0.82rem; border-left: 3px solid #c9d1da; padding-left: 0.6rem; margin-bottom: 1rem;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner='Fitting the surrogates on the 1134 training simulations (about 15 s) ...')
def models():
    return SurrogateSet()


@st.cache_data(show_spinner=False)
def city_data(name):
    return load_city(name)


@st.cache_data(show_spinner=False)
def setpoint_curves(margin, tmax, cm):
    M = models(); rows = []
    for tank in (3, 4):
        for Ta in np.arange(20.0, 50.01, 1.0):
            s = optimise(M, tank, float(Ta), margin, float(tmax), cm)
            if s: rows.append(dict(tank=f'Type {tank}', ambient=Ta, delivery=s['Tpre'], ramp=s['rate'], fill=s['t'], T_end=s['T'], kWh_kg=s['e']))
    return pd.DataFrame(rows)


def axis(title): return alt.Axis(title=title, grid=True, gridColor='#e6eaef', domainColor='#9aa5b1', tickColor='#9aa5b1', labelColor='#4a5562', titleColor='#1d2733')


# ------------------------------------------------------------------ header and sidebar
st.title('H2 Fill Surrogates')
st.markdown('<div class="subtitle">70 MPa hydrogen refuelling of Type 3 and Type 4 tanks: surrogate models of H2FillS simulations, precooling set-points and station energy</div>', unsafe_allow_html=True)
st.markdown('<div class="notice">Research prototype. Reproduces H2FillS simulations of a 99 L tank started at 2 MPa and 25 °C; not validated against measurements; not for operating a station or a vehicle. See the About tab.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Settings')
    margin = st.number_input('Safety margin (°C)', 0.0, 10.0, 2.0, 0.5, help='The end temperature must stay at or below 85 °C minus this margin.')
    tmax = st.number_input('Fill-time limit (s)', 120, 600, 240, 10, help='Maximum time to reach 70 MPa.')
    cm = st.selectbox('Precooler COP model', ['carnot', 'const'], format_func=lambda x: 'Carnot fraction (0.4 × Carnot)' if x == 'carnot' else 'Constant (1.5)',
                      help='Used for the electricity estimates. The Carnot-fraction model takes the condenser at ambient + 10 K.')
    st.divider()
    st.caption('Ranges of the training simulations')
    st.markdown('Ambient 20–50 °C  \nDelivery −40 to +40 °C  \nRamp rate 10–26 MPa/min  \nTanks: Type 3 (Al liner), Type 4 (HDPE liner), 99 L')
    st.divider()
    st.markdown(f'[Source code and data]({REPO})  \nVersion {h2surrogate.__version__} · MIT licence')

M = models()
tab1, tab2, tab3, tab4 = st.tabs(['Predict a fill', 'Set-point finder', 'Station estimate', 'About'])

# ------------------------------------------------------------------ 1. predict
with tab1:
    with st.container(border=True):
        c = st.columns(4)
        tank = c[0].selectbox('Tank', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='p_tank')
        Ta = c[1].number_input('Ambient temperature (°C)', -10.0, 60.0, 35.0, 1.0, key='p_ta')
        Tp = c[2].number_input('Delivery temperature (°C)', -60.0, 60.0, -21.0, 0.5, key='p_tp')
        rate = c[3].number_input('Pressure ramp rate (MPa/min)', 5.0, 40.0, 17.5, 0.5, key='p_r')
    r = M.predict(tank, Ta, Tp, rate, margin)
    for w in r['warnings']:
        st.warning('Extrapolation: ' + w + '. Predictions outside the training range are not reliable.')
    m = st.columns(4)
    m[0].metric('Peak temperature, SVR', f"{r['T_peak_svr']:.1f} °C")
    m[1].metric('Peak temp., Gaussian process', f"{r['T_peak_gp']:.1f} °C", f"± {2 * r['T_peak_gp_std']:.2f} °C (2σ)", delta_color='off')
    m[2].metric('Fill time (to 70 MPa)', f"{r['fill_time']:.0f} s")
    if r['reaches_limit']:
        m[3].metric('At the 85 °C limit', f"{r['SOC_safe']:.1f} % SOC", f"{r['P_safe']:.1f} MPa", delta_color='off')
    else:
        m[3].metric('At the 85 °C limit', 'not reached')
    if r['compliant']:
        st.success(f"Compliant: the predicted end temperature is at most {85 - margin:g} °C (85 °C minus the {margin:g} °C margin).")
    elif r['reaches_limit']:
        st.error('The gas is predicted to reach 85 °C before the end of the fill; a station applying the limit would stop the fill at the SOC and pressure shown.')
    else:
        st.warning(f"Below 85 °C but inside the safety margin ({85 - margin:g} °C).")
    # end temperature against delivery temperature
    xs = np.arange(-40.0, 40.01, 0.5)
    Z = M.features([tank] * len(xs), [Ta] * len(xs), xs, [rate] * len(xs))
    ts = M.svr_T.predict(Z); gm, gs = M.gp_T.predict(Z, return_std=True)
    df = pd.DataFrame(dict(delivery=xs, svr=ts, gp=gm, lo=gm - 2 * gs, hi=gm + 2 * gs))
    base = alt.Chart(df).encode(x=alt.X('delivery:Q', axis=axis('Delivery temperature (°C)'), scale=alt.Scale(domain=[-40, 40])))
    band = base.mark_area(opacity=0.18, color=ORANGE).encode(y='lo:Q', y2='hi:Q')
    l1 = base.mark_line(color=BLUE, strokeWidth=2.4).encode(y=alt.Y('svr:Q', axis=axis('Peak gas temperature (°C)'), scale=alt.Scale(zero=False)))
    l2 = base.mark_line(color=ORANGE, strokeDash=[5, 3], strokeWidth=1.6).encode(y='gp:Q')
    lim = alt.Chart(pd.DataFrame(dict(y=[85.0, 85.0 - margin]), index=['limit', 'target'])).mark_rule(color=GREY, strokeDash=[2, 3]).encode(y='y:Q')
    pt = alt.Chart(pd.DataFrame(dict(x=[Tp], y=[r['T_peak_svr']]))).mark_point(color='#111', filled=True, size=90).encode(x='x:Q', y='y:Q')
    st.altair_chart((band + l1 + l2 + lim + pt).properties(height=300, title=f'End temperature against delivery temperature (Type {tank}, ambient {Ta:g} °C, {rate:g} MPa/min)'), width='stretch')
    st.caption('Solid blue: SVR. Dashed orange: Gaussian process with its ±2σ band. Dotted lines: 85 °C and the target (85 °C minus the margin). Black point: the inputs above. '
               'The Gaussian-process band is the model\'s own posterior spread, not a validated error bound. In the paper, the off-grid error is 0.10 °C for SVR and 0.05 °C for the Gaussian process.')

# ------------------------------------------------------------------ 2. set-point finder
with tab2:
    with st.container(border=True):
        c = st.columns(2)
        tank2 = c[0].selectbox('Tank', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='s_tank')
        Ta2 = c[1].slider('Ambient temperature (°C)', 20.0, 50.0, 35.0, 0.5, key='s_ta')
    s2 = optimise(M, tank2, Ta2, margin, float(tmax), cm)
    if s2 is None:
        st.error('No feasible set-point: reduce the safety margin or raise the fill-time limit.')
    else:
        e0 = float(e_kwh(-40.0, Ta2, cm))
        m = st.columns(5)
        m[0].metric('Delivery temp.', f"{s2['Tpre']:.1f} °C")
        m[1].metric('Ramp rate (MPa/min)', f"{s2['rate']:.1f}")
        m[2].metric('Fill time', f"{s2['t']:.0f} s")
        m[3].metric('End temp. (SVR)', f"{s2['T']:.1f} °C")
        m[4].metric('Electricity (kWh/kg)', f"{s2['e']:.3f}", f"{100 * (s2['e'] / e0 - 1):.0f} % vs −40 °C fill", delta_color='inverse')
        st.caption('The set-point is the lowest-electricity combination whose SVR-predicted end temperature is at most 85 °C minus the margin, within the fill-time limit, searched on a 0.5 °C × 0.5 MPa/min grid. '
                   'In the paper, six such set-points were re-simulated in H2FillS and ended within 0.1 °C of the prediction; the others were not re-simulated.')
    cur = setpoint_curves(margin, tmax, cm)
    col = alt.Color('tank:N', scale=alt.Scale(domain=['Type 3', 'Type 4'], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient='top'))
    a = alt.Chart(cur).mark_line(strokeWidth=2.4, point=True).encode(x=alt.X('ambient:Q', axis=axis('Ambient temperature (°C)')), y=alt.Y('delivery:Q', axis=axis('Delivery temperature (°C)'), scale=alt.Scale(domain=[-45, 45])), color=col).properties(height=260, title='Delivery temperature')
    b = alt.Chart(cur).mark_line(strokeWidth=2.4, point=True).encode(x=alt.X('ambient:Q', axis=axis('Ambient temperature (°C)')), y=alt.Y('ramp:Q', axis=axis('Ramp rate (MPa/min)'), scale=alt.Scale(domain=[10, 26])), color=col).properties(height=260, title='Pressure ramp rate')
    cc = st.columns(2); cc[0].altair_chart(a, width='stretch'); cc[1].altair_chart(b, width='stretch')
    tbl = cur[cur.tank == f'Type {tank2}'].iloc[::2].rename(columns=dict(ambient='Ambient (°C)', delivery='Delivery (°C)', ramp='Ramp (MPa/min)', fill='Fill time (s)', T_end='End temperature (°C)', kWh_kg='Electricity (kWh/kg)')).drop(columns='tank')
    with st.expander(f'Look-up table, Type {tank2}'):
        st.dataframe(tbl.round(3), hide_index=True, width='stretch')
        st.download_button('Download table (CSV)', tbl.round(4).to_csv(index=False), f'setpoints_type{tank2}.csv', 'text/csv')

# ------------------------------------------------------------------ 3. station estimate
with tab3:
    with st.container(border=True):
        c = st.columns(4)
        mode = c[0].radio('Hourly temperatures', ['Built-in city', 'Upload CSV'])
        tank3 = c[1].selectbox('Tank (vehicle fleet)', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='st_tank')
        kg = c[2].number_input('Hydrogen dispensed (kg/day)', 10.0, 20000.0, 100.0, 10.0)
        tariff = c[3].number_input('Electricity tariff (USD/kWh)', 0.0, 1.0, 0.10, 0.005, format='%.3f')
        c2 = st.columns(2)
        h0, h1 = c2[0].slider('Operating hours (local time)', 0, 24, (6, 22))
        hourly = None
        if mode == 'Built-in city':
            city = c2[1].selectbox('City', list(CITIES), index=3)
            hourly = city_data(city)
            c2[1].caption('Hourly 2 m temperature 2005–2024, ERA5 via Open-Meteo (CC BY 4.0), 25 km grid.')
        else:
            up = c2[1].file_uploader('CSV with a time column and a temperature column (°C)', type='csv', help='Open-Meteo exports are accepted as they are.')
            if up is not None:
                hourly = read_temperature_csv(up)
    if hourly is None:
        st.info('Upload a CSV of hourly temperatures to continue.')
    else:
        with st.spinner('Optimising the set-point for each ambient temperature ...'):
            try:
                res = station_estimate(M, hourly, tank3, kg, (h0, h1), tariff, cm, margin, float(tmax))
            except ValueError as e:
                st.error(str(e)); res = None
        if res:
            m = st.columns(4)
            m[0].metric('Electricity, −40 °C fills', f"{res['MWh_yr_S0']:.2f} MWh")
            m[1].metric('Electricity, set-points', f"{res['MWh_yr_S2']:.2f} MWh", f"{-res['saving_pct']:.0f} %", delta_color='inverse')
            m[2].metric('Cost saving per year', f"{res['cost_saving']:,.0f} USD")
            m[3].metric('Peak duty (kW)', f"{res['peak_duty_S0_kW']:.0f} → {res['peak_duty_S2_kW']:.0f}", f"{100 * (res['peak_duty_S2_kW'] / res['peak_duty_S0_kW'] - 1):.0f} %", delta_color='inverse')
            st.markdown(f"Mean set-point: **{res['mean_Tpre']:.1f} °C** at **{res['mean_rate']:.1f} MPa/min**, fill time {res['mean_fill_time']:.0f} s. "
                        f"Operating hours analysed: {res['n_hours']:,}; below 20 °C (treated as 20 °C): {res['hours_below_20_pct']:.0f} %; above 50 °C (treated as 50 °C): {res['hours_above_50_pct']:.1f} %.")
            mo = res['monthly'].melt('month', var_name='strategy', value_name='kWh_kg')
            mo['strategy'] = mo.strategy.map({'S0': '−40 °C, 26 MPa/min', 'S2': 'Surrogate set-points'})
            ch = alt.Chart(mo).mark_bar().encode(x=alt.X('month:O', axis=alt.Axis(title='Month', labelAngle=0, labelColor='#4a5562', titleColor='#1d2733')), xOffset='strategy:N', y=alt.Y('kWh_kg:Q', axis=axis('Precooling electricity (kWh per kg of hydrogen)')),
                                                color=alt.Color('strategy:N', scale=alt.Scale(domain=['−40 °C, 26 MPa/min', 'Surrogate set-points'], range=['#9aa5b1', BLUE]), legend=alt.Legend(title=None, orient='top'))).properties(height=300)
            st.altair_chart(ch, width='stretch')
            summary = pd.DataFrame([dict(metric='MWh/yr, -40 C fills', value=res['MWh_yr_S0']), dict(metric='MWh/yr, surrogate set-points', value=res['MWh_yr_S2']), dict(metric='saving, %', value=res['saving_pct']),
                                    dict(metric='cost saving, USD/yr', value=res['cost_saving']), dict(metric='peak duty S0, kW', value=res['peak_duty_S0_kW']), dict(metric='peak duty S2, kW', value=res['peak_duty_S2_kW'])])
            st.download_button('Download summary (CSV)', summary.round(3).to_csv(index=False), 'station_estimate.csv', 'text/csv')
            st.caption('Electricity only: no capital cost, demand charges or effect of longer fills on throughput. Peak duty is for the hottest hour of the record and one dispenser filling at a time.')

# ------------------------------------------------------------------ 4. about
with tab4:
    a, b = st.columns([3, 2])
    with a:
        st.subheader('Models and data')
        st.markdown('''
The surrogates are fitted to **1134 H2FillS simulations** of a 70 MPa fill of a 99 L tank (2 tanks × 7 ambient temperatures × 9 delivery temperatures × 9 ramp rates),
started at 2 MPa and 25 °C and ended at 70 MPa.

| Target | Model |
|---|---|
| Peak gas temperature | Support-vector regression (RBF) and a Gaussian process |
| State of charge at the 85 °C limit | Multi-layer perceptron |
| Pressure at the 85 °C limit | Multi-layer perceptron |

Hyperparameters are fixed, as in the paper. The fill time is a function of the ramp rate, t = 3.66 s + 4057 / (ramp rate in MPa/min).
Electricity for precooling is e = c<sub>p</sub> (T<sub>amb</sub> − T<sub>pre</sub>) / (3600 COP), with c<sub>p</sub> = 14.3 kJ/(kg K).
''', unsafe_allow_html=True)
        st.subheader('Range of validity')
        st.markdown('Ambient 20–50 °C, delivery −40 to +40 °C, ramp rate 10–26 MPa/min, one 99 L tank started at 2 MPa and 25 °C. Inputs outside these ranges are extrapolation; tree ensembles under-predicted by 2.5 °C there in the paper, and kernel methods are not safe either.')
        st.subheader('What this is not')
        st.markdown('It is not validated against measured fills, ignores initial tank state, tank volume, humidity and station hardware, and must not be used to operate a station or a vehicle.')
    with b:
        st.subheader('Accuracy reported in the paper')
        st.markdown("""
| Peak-temperature model | RMSE (°C) | False-safe (of 131) |
|---|---|---|
| SVR | 0.10 | 0 |
| Gaussian process | 0.05 | 0 |
| XGBoost, RF, LightGBM | about 3 | 10–11 |
""")
        st.caption('223 off-grid simulations; 131 of them exceed 85 °C. Six surrogate-selected set-points re-simulated in H2FillS ended within 0.1 °C of the SVR prediction.')
        st.subheader('Cite')
        st.markdown(f'AitAkka N\'alla A, El Alami M, Gounni A. Machine-learning surrogates for the safety-limited end state of hydrogen refueling in Type 3 and Type 4 tanks (submitted).  \nSoftware: [{REPO}]({REPO}), version {h2surrogate.__version__}, MIT licence. Hourly temperatures: ERA5 via Open-Meteo, CC BY 4.0.')
