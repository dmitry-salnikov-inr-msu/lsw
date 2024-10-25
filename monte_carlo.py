import numpy as np

import pygsl._numobj as numx
import pygsl
import pygsl.rng
import pygsl.monte as monte

r = pygsl.rng.mt19937_1999()

M_PI = numx.pi


def monte_carlo(ndim,
                _ndim_integrand,
                params,
                lower,
                upper,
                calls):
    G = monte.gsl_monte_function(_ndim_integrand, params, ndim)
    s = monte.plain(ndim)
    s.init()
    res, abserr = s.integrate(G, lower, upper, calls, r)
    return np.array([res, abserr])
