import numpy as np

CP_H2 = 14.3  # kJ/(kg K)


def cop(T_pre, T_amb, model='carnot'):
    """Coefficient of performance of the precooler. 'const' = 1.5; 'carnot' = 0.4 x Carnot, with the condenser at T_amb + 10 K."""
    if model == 'const':
        return 1.5 * np.ones(np.broadcast(T_pre, T_amb).shape)
    Tc = np.asarray(T_pre, float) + 273.15
    Th = np.asarray(T_amb, float) + 10.0 + 273.15
    with np.errstate(divide='ignore', invalid='ignore'):
        return 0.4 * Tc / (Th - Tc)


def e_kwh(T_pre, T_amb, model='carnot'):
    """Electricity (kWh) to cool 1 kg of hydrogen from T_amb to T_pre."""
    lift = np.maximum(np.asarray(T_amb, float) - np.asarray(T_pre, float), 0.0)
    return CP_H2 * lift / (3600.0 * cop(T_pre, T_amb, model))


def fill_time(rate, a=3.66, b=4057.0):
    """Fill time (s) at a pressure ramp rate in MPa/min (coefficients of the paper)."""
    return a + b / np.asarray(rate, float)
