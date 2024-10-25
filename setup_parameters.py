import importlib
import numpy as np
import dimensional_variables as dv

DimensionalVariable = dv.DimensionalVariable
ValueWithError = dv.ValueWithError

pump_mode_amplitude_SRF = DimensionalVariable(
    array=0.1,
    dimension='Tesla',
)

pump_mode_amplitude_NRF = DimensionalVariable(
    array=1.,
    dimension='Tesla',
)

external_magnetic_field = DimensionalVariable(
    array=3,
    dimension='Tesla',
)

time = DimensionalVariable(
    array=60*60*24*365,
    dimension='second',
)

temperature = DimensionalVariable(
    array=4,
    dimension='Kelvin',
)

quality_factor_SRF = 1e+10

quality_factor_NRF = 1e+5

Signal_to_Noise_Ratio = 5
