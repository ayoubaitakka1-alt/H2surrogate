"""h2surrogate: machine-learning surrogates of 70 MPa hydrogen fills (H2FillS), set-point selection and station energy estimates.

Research code accompanying the manuscript. It reproduces H2FillS simulations of one 99 L tank (Type 3 or Type 4) started at
2 MPa and 25 degC. It has not been validated against measurements and is not a safety tool.
"""
from .models import SurrogateSet, load_tables, FEATS, LIMITS
from .energy import e_kwh, cop, CP_H2, fill_time
from .optimise import optimise
from .station import station_estimate, load_city, CITIES

__all__ = ['SurrogateSet', 'load_tables', 'FEATS', 'LIMITS', 'e_kwh', 'cop', 'CP_H2', 'fill_time', 'optimise',
           'station_estimate', 'load_city', 'CITIES']
__version__ = '1.0.0'
