# Data

| File | Content |
|---|---|
| `train_table.csv` | 1134 H2FillS fills on the factorial grid (2 tanks x 7 ambient 20-50 degC x 9 delivery -40..+40 degC x 9 ramp 10-26 MPa/min) |
| `val_table.csv` | 223 off-grid fills used for validation (131 exceed 85 degC) |
| `confirmation_table.csv` | 16 Type 3 fills with all inputs off-grid |
| `setpoint_verification_runs.csv`, `verification_results.csv`, `verification_runs/SP*.xlsx` | the six surrogate-selected set-points re-simulated in H2FillS, with predictions and results |
| `raw/` | the original H2FillS result files (training and validation sets, confirmation runs as a zip) |
| `climate/*.csv.gz` | hourly 2 m air temperature 2005-2024 (ERA5 via Open-Meteo, CC BY 4.0; Zippenfenig, Open-Meteo.com, doi:10.5281/zenodo.7970649; Hersbach et al. 2020, doi:10.1002/qj.3803), local time |

Table columns: `tank` (3 or 4), `T_amb`, `T_pre` (delivery temperature, degC), `rate` (MPa/min), `T_end`, `SOC_end`, `P_end`, `t_end` (end of fill),
`T_peak` (peak bulk gas temperature, equal to the final value because the temperature rises monotonically), `TL_peak` (peak liner temperature), `m_H2` (kg),
`Q_pre` (kJ), `crossed` (1 if the gas reached 85 degC before the end), `t85`, `SOC_safe`, `P_safe` (SOC and pressure at the first crossing of 85 degC, or the end state if it never crosses).
H2FillS settings: Cv equation "H2FillS Original", initial tank state 2 MPa and 25 degC, terminating pressure 70 MPa.
