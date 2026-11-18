"""Streamlit app for the hydrogen-fill surrogates. Run:  streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st
from h2surrogate import SurrogateSet, optimise, station_estimate, load_city, CITIES, LIMITS
from h2surrogate.station import read_temperature_csv

st.set_page_config(page_title='H2 fill surrogates', layout='wide')


@st.cache_resource(show_spinner='Fitting the surrogates on the 1134 training simulations (about 15 s) ...')
def models():
    return SurrogateSet()


@st.cache_data(show_spinner=False)
def city_data(name):
    return load_city(name)


st.title('Hydrogen fill surrogates (70 MPa, 99 L tank)')
st.caption('Research prototype: reproduces H2FillS simulations (99 L tank, 2 MPa and 25 °C start); not validated against measurements; not for operating a station or vehicle. See the About tab.')
M = models()
tab1, tab2, tab3, tab4 = st.tabs(['Predict a fill', 'Set-point finder', 'Station estimate', 'About'])

with tab1:
    c = st.columns(5)
    tank = c[0].selectbox('Tank', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='p_tank')
    Ta = c[1].number_input('Ambient (°C)', -10.0, 60.0, 35.0, 1.0, key='p_ta')
    Tp = c[2].number_input('Delivery temperature (°C)', -60.0, 60.0, -21.0, 0.5, key='p_tp')
    rate = c[3].number_input('Ramp rate (MPa/min)', 5.0, 40.0, 17.5, 0.5, key='p_r')
    margin = c[4].number_input('Safety margin (°C)', 0.0, 10.0, 2.0, 0.5, key='p_m')
    r = M.predict(tank, Ta, Tp, rate, margin)
    for w in r['warnings']:
        st.warning('Extrapolation: ' + w)
    m = st.columns(4)
    m[0].metric('Peak gas temperature (SVR)', f"{r['T_peak_svr']:.1f} °C")
    m[1].metric('Peak gas temperature (Gaussian process)', f"{r['T_peak_gp']:.1f} °C", f"± {2 * r['T_peak_gp_std']:.2f} (2σ, model-internal)", delta_color='off')
    m[2].metric('Fill time to 70 MPa', f"{r['fill_time']:.0f} s")
    if r['reaches_limit']:
        m[3].metric('At the 85 °C limit (MLP)', f"{r['SOC_safe']:.1f} % SOC", f"{r['P_safe']:.1f} MPa", delta_color='off')
    else:
        m[3].metric('At the 85 °C limit', 'not reached')
    if r['compliant']:
        st.success(f"Predicted end temperature is at most 85 − {margin:g} = {85 - margin:g} °C.")
    elif r['reaches_limit']:
        st.error('Predicted to reach 85 °C before the end of the fill; the fill would be stopped at the SOC shown.')
    else:
        st.warning(f"Below 85 °C but inside the safety margin ({85 - margin:g} °C).")
    st.caption('The two temperature models agree within a few tenths of a degree inside the training range (off-grid RMSE 0.10 °C SVR, 0.05 °C Gaussian process in the paper). '
               'The standard deviation is the Gaussian-process posterior, not a validated error bound.')

with tab2:
    c = st.columns(5)
    tank2 = c[0].selectbox('Tank', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='s_tank')
    Ta2 = c[1].slider('Ambient (°C)', 20.0, 50.0, 35.0, 0.5, key='s_ta')
    mg = c[2].number_input('Safety margin (°C)', 0.0, 10.0, 2.0, 0.5, key='s_m')
    tmax = c[3].number_input('Fill-time limit (s)', 120, 600, 240, 10, key='s_t')
    cm = c[4].selectbox('COP model', ['carnot', 'const'], format_func=lambda x: 'Carnot fraction (0.4)' if x == 'carnot' else 'Constant 1.5', key='s_c')
    s2 = optimise(M, tank2, Ta2, mg, float(tmax), cm)
    if s2 is None:
        st.error('No feasible set-point: relax the margin or raise the fill-time limit.')
    else:
        e0 = float(__import__('h2surrogate').e_kwh(-40.0, Ta2, cm))
        m = st.columns(5)
        m[0].metric('Delivery temperature', f"{s2['Tpre']:.1f} °C")
        m[1].metric('Ramp rate', f"{s2['rate']:.1f} MPa/min")
        m[2].metric('Fill time', f"{s2['t']:.0f} s")
        m[3].metric('Predicted end temperature', f"{s2['T']:.1f} °C")
        m[4].metric('Precooling electricity', f"{s2['e']:.3f} kWh/kg", f"{100 * (s2['e'] / e0 - 1):.0f} % vs −40 °C fill ({e0:.3f})", delta_color='inverse')
        st.caption('Set-points are the lowest-electricity combination that keeps the SVR-predicted end temperature at or below 85 °C minus the margin. '
                   'Six of them were re-simulated in H2FillS in the paper and ended within 0.1 °C of the prediction; the others were not. Electricity is a first-order estimate that depends on the COP model.')
    amb = np.arange(20, 50.1, 2.0)
    tab = []
    for Tt in amb:
        s = optimise(M, tank2, float(Tt), mg, float(tmax), cm)
        tab.append(dict(ambient=Tt, delivery_C=None if s is None else s['Tpre'], ramp_MPa_min=None if s is None else s['rate'], fill_s=None if s is None else round(s['t'], 1),
                        T_end_C=None if s is None else round(s['T'], 1), kWh_per_kg=None if s is None else round(s['e'], 4)))
    tab = pd.DataFrame(tab)
    st.subheader('Look-up table for this tank')
    st.dataframe(tab, hide_index=True, width='stretch')
    st.download_button('Download table (CSV)', tab.to_csv(index=False), f'setpoints_type{tank2}.csv', 'text/csv')

with tab3:
    c = st.columns(4)
    mode = c[0].radio('Hourly temperatures', ['Built-in city', 'Upload CSV'], horizontal=False)
    tank3 = c[1].selectbox('Tank', [3, 4], index=1, format_func=lambda t: f'Type {t}', key='st_tank')
    kg = c[2].number_input('Hydrogen dispensed (kg/day)', 10.0, 20000.0, 100.0, 10.0)
    tariff = c[3].number_input('Electricity tariff (USD/kWh)', 0.0, 1.0, 0.10, 0.005, format='%.3f')
    c2 = st.columns(5)
    h0, h1 = c2[0].slider('Operating hours', 0, 24, (6, 22))
    cm3 = c2[1].selectbox('COP model', ['carnot', 'const'], format_func=lambda x: 'Carnot fraction (0.4)' if x == 'carnot' else 'Constant 1.5', key='st_c')
    mg3 = c2[2].number_input('Safety margin (°C)', 0.0, 10.0, 2.0, 0.5, key='st_m')
    tm3 = c2[3].number_input('Fill-time limit (s)', 120, 600, 240, 10, key='st_t')
    hourly = None
    if mode == 'Built-in city':
        city = c2[4].selectbox('City', list(CITIES), index=3)
        hourly = city_data(city)
        st.caption('Hourly 2 m temperature 2005–2024, ERA5 reanalysis via Open-Meteo (CC BY 4.0), local time, 25 km grid.')
    else:
        up = st.file_uploader('CSV with a time column and a temperature column (°C); Open-Meteo exports are accepted', type='csv')
        if up is not None:
            hourly = read_temperature_csv(up)
    if hourly is not None:
        with st.spinner('Optimising the set-point for each ambient temperature ...'):
            try:
                res = station_estimate(M, hourly, tank3, kg, (h0, h1), tariff, cm3, mg3, float(tm3))
            except ValueError as e:
                st.error(str(e)); res = None
        if res:
            m = st.columns(4)
            m[0].metric('Annual electricity, −40 °C fills', f"{res['MWh_yr_S0']:.2f} MWh")
            m[1].metric('Annual electricity, surrogate set-points', f"{res['MWh_yr_S2']:.2f} MWh", f"{-res['saving_pct']:.0f} %", delta_color='inverse')
            m[2].metric('Annual cost saving', f"{res['cost_saving']:,.0f} USD")
            m[3].metric('Peak precooler duty (S0 → S2)', f"{res['peak_duty_S0_kW']:.0f} → {res['peak_duty_S2_kW']:.0f} kW")
            st.write(f"Mean set-point: {res['mean_Tpre']:.1f} °C at {res['mean_rate']:.1f} MPa/min, fill time {res['mean_fill_time']:.0f} s. "
                     f"Hours below 20 °C (treated as 20 °C): {res['hours_below_20_pct']:.0f} %; above 50 °C (treated as 50 °C): {res['hours_above_50_pct']:.1f} %.")
            mo = res['monthly'].set_index('month').rename(columns={'S0': '−40 °C fills', 'S2': 'surrogate set-points'})
            st.bar_chart(mo, y_label='kWh per kg of hydrogen', x_label='Month')
            st.caption('Peak duty is for the hottest hour of the record and one dispenser. Electricity only; capital cost, demand charges and the effect of longer fills on throughput are not included.')

with tab4:
    st.markdown('''
**What this is.** Surrogates fitted to 1134 H2FillS simulations (2 tanks × 7 ambient temperatures × 9 delivery temperatures × 9 ramp rates) of a 70 MPa fill of a 99 L tank
started at 2 MPa and 25 °C, ending at 70 MPa. Models: support-vector regression and a Gaussian process for the peak gas temperature, and a multi-layer perceptron for the
state of charge and pressure at the 85 °C limit.

**Range of validity.** Ambient 20–50 °C, delivery −40 to +40 °C, ramp rate 10–26 MPa/min, one 99 L tank started at 2 MPa and 25 °C.

**What it is not.** It is not validated against measured fills, it ignores initial tank state, volume, humidity and station hardware, and it must not be used to operate a
station or a vehicle. Inputs outside the training range are extrapolation (tree ensembles in the paper under-predicted by 2.5 °C there; kernel methods do better but are not safe either).

**Paper and code.** See the README of the repository for the citation, the data, and how to reproduce the tables and figures.
''')
