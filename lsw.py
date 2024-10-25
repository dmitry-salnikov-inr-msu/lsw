from cubature import cubature
import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import optimize
from tqdm import tqdm
from IPython.display import display, Latex

import dimensional_variables as dv
import setup_parameters as sp
import monte_carlo as mc


ValueWithError = dv.ValueWithError
DimensionalVariable = dv.DimensionalVariable
ValueWithError_from_array = dv.ValueWithError_from_array
DimensionalVariable_from_array = dv.DimensionalVariable_from_array

pump_mode_amplitude_SRF = sp.pump_mode_amplitude_SRF
pump_mode_amplitude_NRF = sp.pump_mode_amplitude_NRF
external_magnetic_field = sp.external_magnetic_field
time = sp.time
temperature = sp.temperature
quality_factor_SRF = sp.quality_factor_SRF
quality_factor_NRF = sp.quality_factor_NRF
Signal_to_Noise_Ratio = sp.Signal_to_Noise_Ratio

M_PI = mc.M_PI
monte_carlo = mc.monte_carlo


class lsw:
    def __init__(self,
                 particles_type: str,
                 cavity_mode: str,
                 setup_geometry: str,
                 cavity_sizes: list,
                 masses: DimensionalVariable,
                 relerr: np.float64 = np.float64(1e-3),
                 calls: int = 100_000,
                 number_of_6d_points: int = 10,
                 show_image=False) -> None:

        self._show_image = show_image

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": 'monospace',
            "font.serif": ['Computer Modern Typewriter'],
            "font.size": 14
        })
        assert particles_type in ('DP', 'ALP'), \
            f'particles type {particles_type} is not available'
        self.particles_type = particles_type
        assert cavity_mode in ('TM010', 'TE011'), \
            f'cavity mode {cavity_mode} is not available'
        self.cavity_mode = cavity_mode
        assert setup_geometry in ('separated', 'enveloped'), \
            f'setup_geometry {setup_geometry} is not available'
        self.setup_geometry = setup_geometry
        assert len(cavity_sizes) == 3, \
            f'the cavity_sizes list must contain' + \
            f'3 variables: R, L, d; not {len(cavity_sizes)}'
        self.R, self.L, self.d = cavity_sizes
        self.relerr = np.float64(relerr)
        self.masses = masses
        self.len = masses.array.shape[0]
        self.number_of_6d_points = number_of_6d_points
        self.step_of_6d_points = self.len // self.number_of_6d_points
        self.calls = calls
        match self.cavity_mode:
            case 'TM010':
                self.zeros = lambda n: special.jn_zeros(0, n)[-1]
                self.x = self.zeros(1)
                self.prho = self.x / self.R
                self.pz = 0.
                self.omega = self.prho
                self.J = lambda x: special.jv(0, x)
                self.Jp = lambda x: special.jvp(0, x)
                self.Y = lambda x: special.yv(0, x)
                self.Yp = lambda x: special.yvp(0, x)
                self.H = lambda x: special.hankel1(0, x)
                self.Hp = lambda x: -special.hankel1(1, x)
            case 'TE011':
                self.zeros = lambda n: special.jnp_zeros(0, n)[-1]
                self.x = self.zeros(1)
                self.prho = self.x / self.R
                self.pz = np.pi / self.L
                self.omega = np.sqrt(self.prho**2 + self.pz**2)
                self.J = lambda x: special.jv(1, x)
                self.Jp = lambda x: special.jvp(1, x)
                self.Y = lambda x: special.yv(1, x)
                self.Yp = lambda x: special.yvp(1, x)
                self.H = lambda x: special.hankel1(1, x)
                self.Hp = lambda x: special.hankel1(0, x) / 2 - \
                    special.hankel1(2, x) / 2
        self.JH = lambda x1, x2: (np.exp(1j * (x2 - x1)) /
                                  np.pi / np.sqrt(x1 * x2))
        match self.setup_geometry:
            case 'separated':
                self.V1 = np.pi * self.R**2 * self.L
                self.V2 = np.pi * self.R**2 * self.L
                self.alpha1 = 1. / np.abs(self.Jp(self.prho * self.R))
                self.alpha2 = 1. / np.abs(self.Jp(self.prho * self.R))
                if self.cavity_mode == 'TE011':
                    self.alpha1 *= np.sqrt(2)
                    self.alpha2 *= np.sqrt(2)
                if self._show_image:
                    self._CheckingBoundaryConditions()
                self.lower6d = [0., 0., 0., 0., 0., 0.]
                self.upper6d = [1., 1., 2 * M_PI, 2 * M_PI, 1., 1.]
                if self.cavity_mode == 'TM010' and self.particles_type == 'DP':
                    self.lower4d = [0., 0., 0., 0.]
                    self.upper4d = [1., 1., 2 * M_PI, 2 * M_PI]
            case 'enveloped':
                self.R1 = self.R
                self.R2 = self.R + self.d
                self.Z = lambda x: (self.J(x) -
                                    self.Y(x) * self.J(self.prho * self.R2) /
                                    self.Y(self.prho * self.R2))
                self.Zp = lambda x: (self.Jp(x) -
                                     self.Yp(x) * self.J(self.prho * self.R2) /
                                     self.Y(self.prho * self.R2))
                self.R3 = self._CalculateR3()
                self.V1 = np.pi * self.R1**2 * self.L
                self.V2 = np.pi * (self.R3**2 - self.R2**2) * self.L
                self.alpha1 = 1. / np.abs(self.Jp(self.prho * self.R1))
                self.alpha2 = np.sqrt((self.R3**2 - self.R2**2) /
                                      (self.R3**2 *
                                       self.Zp(self.prho * self.R3)**2 -
                                       self.R2**2 *
                                       self.Zp(self.prho * self.R2)**2))
                if self.cavity_mode == 'TE011':
                    self.alpha1 *= np.sqrt(2)
                    self.alpha2 *= np.sqrt(2)
                if self._show_image:
                    self._CheckingBoundaryConditions()
                self.lower6d = [0., self.R2 / self.R1, 0., 0., 0., 0.]
                self.upper6d = [
                    1., self.R3 / self.R1, 2 * M_PI, 2 * M_PI, 1., 1.]
        self._CalculatePreFactor()
        self._AxialFunction = self._AxialFunctions(
        )[self.setup_geometry][self.cavity_mode]
        self._RadialFunction = self._RadialFunctions(
        )[self.setup_geometry][self.cavity_mode][self.particles_type]
        self._1d_integrand = self._1d_integrands(
        )[self.setup_geometry]
        self._AsymptoticFormFactor = self._AsymptoticFormFactors(
        )[self.setup_geometry][self.cavity_mode][self.particles_type]
        self._6d_integrand = self._6d_integrands(
        )[self.setup_geometry][self.cavity_mode][self.particles_type]

    def _CalculateR3(self) -> np.float64:
        start_point = self.prho * self.R2 + np.pi
        out = self.prho * self.R2
        while np.abs(out - self.prho * self.R2) < np.pi / 2:
            out = optimize.fsolve(self.Z, start_point)[0]
            start_point += np.pi / 2
        out /= self.prho
        return out

    def _CheckingBoundaryConditions(self) -> None:
        match self.setup_geometry:
            case 'separated':
                display(Latex(r'Separated geometry'))
                match self.cavity_mode:
                    case 'TM010':
                        label = [r'${\cal E}_{z}$']
                        display(Latex(r'${\rm TM}_{010}-{\rm mode}$'))
                    case 'TE011':
                        label = [r'${\cal E}_{\varphi}$']
                        display(Latex(r'${\rm TE}_{011}-{\rm mode}$'))
                display(Latex(fr'$L = {self.L:.2f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$R = {self.R:.2f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$d = {int(1e+6*self.d)}$' +
                              r' $\mu\rm{m};$'))
                display(Latex(fr'$J(p_\rho R) =$' +
                              fr' $\tt{self.J(self.prho*self.R):.3e}$;'))
                display(Latex(fr"$J'(p_\rho R) =$" +
                              fr' ${self.Jp(self.prho*self.R):.2f}$.'))
                if np.abs(self.J(self.prho * self.R)) < 1e-10:
                    display(Latex('The boundary conditions are met.'))
                else:
                    display(Latex('The boundary conditions are not met.'))
                x = np.linspace(0, self.R, 1000)
                y = self.J(self.prho * x)
                y = self.L * y / y.max() / 2.5
                fig = plt.figure(figsize=(7, 5))
                ax = plt.subplot()
                ax.plot(x, y,
                        color='blue',
                        linewidth=3)
                ax.legend(
                    label,
                    loc='center',
                    bbox_to_anchor=(
                        0.15,
                        0.12),
                    shadow=True,
                    framealpha=1,
                    facecolor='whitesmoke')
                ax.set_xlabel(r'$\rho \ [\rm{m}]$')
                ax.set_ylabel(r'$z \ [\rm{m}]$')
                ax.vlines((self.R,), ymin=-self.L / 2,
                          ymax=self.L / 2, color='0', linewidth=4)
                ax.vlines((0,), ymin=-self.L / 2, ymax=self.L / 2,
                          color='black', linewidth=2, linestyle='--')
                ax.hlines((-self.L / 2, self.L / 2), xmin=0,
                          xmax=self.R, color='black', linewidth=4)
                ax.hlines((0,), xmin=0, xmax=self.R,
                          color='black', linewidth=2, linestyle='--')
                ax.grid(True,
                        which='major',
                        ls='dashed',
                        linewidth=0.5,
                        color='black')
                plt.show()
            case 'enveloped':
                display(Latex(r'Enveloped geometry'))
                match self.cavity_mode:
                    case 'TM010':
                        legend = r'${\cal E}_{z}$'
                        display(Latex(r'${\rm TM}_{010}-{\rm mode}$'))
                    case 'TE011':
                        legend = r'${\cal E}_{\varphi}$'
                        display(Latex(r'${\rm TE}_{011}-{\rm mode}$'))
                display(Latex(fr'$L = {self.L:.5f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$R_1 = {self.R1:.5f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$R_2 = {self.R2:.5f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$R_3 = {self.R3:.5f}$' +
                              r' $\rm{m};$'))
                display(Latex(fr'$d = {int(1e+6 * self.d)}$' +
                              r' $\mu\rm{m};$'))
                print()
                display(Latex(fr'$J(p_\rho R_1) =$' +
                              fr' $\tt{self.J(self.prho * self.R1):.3e}$;'))
                display(Latex(fr"$Z(p_\rho R_2) =$" +
                              fr' $\tt{self.Z(self.prho * self.R2):.3e}$;'))
                display(Latex(fr"$Z(p_\rho R_3) =$" +
                              fr' $\tt{self.Z(self.prho * self.R3):.3e}$;'))
                print()
                display(Latex(fr"$J'(p_\rho R_1) =$" +
                              fr' ${self.Jp(self.prho * self.R1):.2f}$;'))
                display(Latex(fr"$Z'(p_\rho R_2) =$" +
                              fr' ${self.Zp(self.prho * self.R2):.2f}$;'))
                display(Latex(fr"$Z'(p_\rho R_3) =$" +
                              fr' ${self.Zp(self.prho * self.R3):.2f}$;'))
                print()
                value = self.prho * self.R3 * \
                    self.Zp(self.prho * self.R3) * \
                    self.J(self.prho * self.R3) - \
                    self.prho * self.R2 * \
                    self.Zp(self.prho * self.R2) * \
                    self.J(self.prho * self.R2)
                display(
                    Latex(
                        fr"$p_\rho R_3 Z'(p_\rho R_3)J(p_\rho R_3) - $" +
                        fr"$p_\rho R_2 Z'(p_\rho R_2)J(p_\rho R_2) = $" +
                        fr"$\tt{value:3e}$;"))
                value = self.prho * self.R3 * \
                    self.Zp(self.prho * self.R3) * \
                    self.Y(self.prho * self.R3) - \
                    self.prho * self.R2 * \
                    self.Zp(self.prho * self.R2) * \
                    self.Y(self.prho * self.R2)
                display(
                    Latex(
                        fr"$p_\rho R_3 Z'(p_\rho R_3)Y(p_\rho R_3) - $" +
                        fr"$p_\rho R_2 Z'(p_\rho R_2)Y(p_\rho R_2) = $" +
                        fr"$\tt{value:3e}$;"))
                print()
                if np.abs(self.J(self.prho * self.R1)) < 1e-10 and \
                   np.abs(self.Z(self.prho * self.R2)) < 1e-10 and \
                   np.abs(self.Z(self.prho * self.R3)) < 1e-10:
                    display(Latex('The boundary conditions are met.'))
                else:
                    display(Latex('The boundary conditions are not met.'))
                x1 = np.linspace(0., self.R1, 1000)
                y1 = self.J(self.prho * x1)
                x2 = np.linspace(self.R2, self.R3, 1000)
                y2 = self.Z(self.prho * x2)
                y2 = self.L * y2 / y1.max() / 2.5
                y1 = self.L * y1 / y1.max() / 2.5
                fig = plt.figure(figsize=(7, 5))
                ax = plt.subplot()
                ax.plot(x1, y1,
                        color='blue',
                        linewidth=3)
                ax.plot(x2, y2,
                        color='red',
                        linewidth=3)
                labels = [r'$J(x)$', r'$Z(x)$']
                ax.legend(
                    labels,
                    loc='center',
                    bbox_to_anchor=(
                        0.17,
                        0.16),
                    shadow=True,
                    framealpha=1,
                    facecolor='whitesmoke')
                ax.set_xlabel(r'$\rho \ [\rm{m}]$')
                ax.set_ylabel(r'$z \ [\rm{m}]$')
                ax.vlines((self.R1, self.R2, self.R3,),
                          ymin=-self.L / 2, ymax=self.L / 2,
                          color='black', linewidth=4)
                ax.vlines((0,), ymin=-self.L / 2, ymax=self.L / 2,
                          color='black', linewidth=2, linestyle='--')
                ax.hlines((-self.L / 2, self.L / 2), xmin=0,
                          xmax=self.R1, color='black', linewidth=4)
                ax.hlines((-self.L / 2, self.L / 2), xmin=self.R2,
                          xmax=self.R3, color='black', linewidth=4)
                ax.hlines((0,), xmin=0, xmax=self.R3,
                          color='black', linewidth=2, linestyle='--')
                ax.grid(True,
                        which='major',
                        ls='dashed',
                        linewidth=0.5,
                        color='black')
                plt.show()

    def _CalculatePreFactor(self) -> None:
        out = (np.pi * self.alpha1 * self.alpha2 *
               self.R**4 * self.L**2 / (self.V1 * self.V2 * self.omega))
        match self.particles_type:
            case 'ALP':
                if self.cavity_mode == 'TE011':
                    out *= 0.5 * 0.5**2
            case 'DP':
                out /= self.omega**2
                match self.cavity_mode:
                    case 'TM010':
                        out *= 1. / self.R**2
                    case 'TE011':
                        out *= 0.5**2 * self.masses.array.value**2
        out *= np.ones(self.len)
        out1d = DimensionalVariable(array=out)
        out6d = DimensionalVariable(
            array=out[::self.step_of_6d_points] / np.pi)

        self.FormFactor = {'1d-integration': out1d,
                           '6d-integration': out6d,
                           'right_asymptotic': out1d}

    def _AxialFunction_TM010_separated(self,
                                       q: np.float64) -> np.complex128:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out = ((np.exp(0.5 * 1j * (2 * self.L + self.d) * q) -
                        np.exp(0.5 * 1j * self.d * q)) /
                       (1j * q * self.L))**2 / q
            except Warning:
                out = np.float64(0)

        return out

    def _AxialFunction_TM010_enveloped(self,
                                       kz: np.float64) -> np.float64:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out = (np.sin(0.5 * kz * self.L) /
                       (0.5 * kz * self.L))**2
            except Warning:
                out = np.float64(1)

        return out

    def _AxialFunction_TE011_separated(self,
                                       q: np.complex128) -> np.complex128:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                out = np.float64(1) / q
            except Warning:
                return np.float64(0)

            try:
                out *= (0.5 * np.pi *
                        (np.exp(0.5 * 1j * (2 * self.L + self.d) * q) +
                         np.exp(0.5 * 1j * self.d * q)) /
                        ((0.5 * q * self.L)**2 -
                         (0.5 * self.pz * self.L)**2))**2
            except Warning:
                out *= np.exp(1j * q * (self.L + self.d)) / q

        return out

    def _AxialFunction_TE011_enveloped(self,
                                       kz: np.float64) -> np.float64:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out = (np.pi * np.cos(0.5 * kz * self.L) /
                       ((0.5 * kz * self.L)**2 -
                        (0.5 * self.pz * self.L)**2))**2
            except Warning:
                out = np.float64(1)

        return out

    def _AxialFunctions(self):
        return {
            'separated':
            {'TM010': lambda q:
             self._AxialFunction_TM010_separated(q),
             'TE011': lambda q:
             self._AxialFunction_TE011_separated(q)},
            'enveloped':
            {'TM010': lambda kz:
             self._AxialFunction_TM010_enveloped(kz),
             'TE011': lambda kz:
             self._AxialFunction_TE011_enveloped(kz)},
        }

    def _RadialFunction_separated(self,
                                  krho: np.float64,
                                  degree: int) -> np.float64:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                temp = (self.prho * self.R * self.Jp(self.prho * self.R) *
                        self.J(krho * self.R) /
                        ((krho * self.R)**2 - (self.prho * self.R)**2))
            except Warning:
                temp = (self.prho * self.R * self.Jp(self.prho * self.R) *
                        self.Jp(krho * self.R) /
                        (2. * krho * self.R))

        out = (krho * temp**2 *
               ((krho * self.R)**2 - (self.prho * self.R)**2)**degree)

        return out

    def _RadialFunction_enveloped(self,
                                  q: np.complex128,
                                  degree: int) -> np.complex128:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out3 = self.J(q * self.R1) * self.H(q * self.R3)
            except Warning:
                out3 = self.JH(q * self.R1, q * self.R3)

            out3 *= self.prho * self.R1 * self.Jp(self.prho * self.R1)
            out3 *= self.prho * self.R3 * self.Zp(self.prho * self.R3)

            try:
                out2 = self.J(q * self.R1) * self.H(q * self.R2)
            except Warning:
                out2 = self.JH(q * self.R1, q * self.R2)

            out2 *= self.prho * self.R1 * self.Jp(self.prho * self.R1)
            out2 *= self.prho * self.R2 * self.Zp(self.prho * self.R2)

            try:
                out = (out3 - out2) / ((q * self.R1)**2 -
                                       (self.prho * self.R1)**2)**2
            except Warning:
                out3 = (self.prho * self.R1 * self.Jp(self.prho * self.R1) *
                        self.prho * self.R3 * self.Zp(self.prho * self.R3) *
                        self.R1 * self.R3 *
                        self.Jp(q * self.R1) * self.Hp(q * self.R3) /
                        (2 * self.prho * self.R1**2)**2)

                out2 = (self.prho * self.R1 * self.Jp(self.prho * self.R1) *
                        self.prho * self.R2 * self.Zp(self.prho * self.R2) *
                        self.R1 * self.R2 *
                        self.Jp(q * self.R1) * self.Hp(q * self.R2) /
                        (2 * self.prho * self.R1**2)**2)

                out = out3 - out2

        out *= ((q * self.R1)**2 - (self.prho * self.R1)**2)**degree

        return out

    def _RadialFunctions(self):
        return {'separated':
                {'TM010':
                 {'DP': lambda krho: self._RadialFunction_separated(krho, 1),
                  'ALP': lambda krho: self._RadialFunction_separated(krho, 0)},
                 'TE011':
                 {'DP': lambda krho: self._RadialFunction_separated(krho, 0),
                  'ALP': lambda krho: self._RadialFunction_separated(krho, 0)},
                 },
                'enveloped':
                {'TM010':
                 {'DP': lambda q: self._RadialFunction_enveloped(q, 1),
                  'ALP': lambda q: self._RadialFunction_enveloped(q, 0)},
                 'TE011':
                 {'DP': lambda q: self._RadialFunction_enveloped(q, 0),
                  'ALP': lambda q: self._RadialFunction_enveloped(q, 0)},
                 }
                }

    def _1d_integrand_separated(self,
                                k: np.complex128,
                                t_array: np.ndarray) -> np.complex128:

        t, = t_array

        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                krho = (np.abs(k) + t) / (1 - t)
                J = (np.abs(k) + 1) / (1 - t)**2
            except Warning:
                krho = np.float64(0)
                J = np.float64(0)

        q = np.sqrt(k**2 - krho**2)

        out = J * self._RadialFunction(krho) * self._AxialFunction(q)

        krho = np.abs(k) * t
        J = np.abs(k)
        q = np.sqrt(k**2 - krho**2)

        out += J * self._RadialFunction(krho) * self._AxialFunction(q)

        return out

    def _1d_integrand_enveloped(self,
                                k: np.complex128,
                                t_array: np.ndarray) -> np.complex128:

        t, = t_array

        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                kz = (np.abs(k) + t) / (1 - t)
                J = (np.abs(k) + 1) / (1 - t)**2
            except Warning:
                kz = np.float64(0)
                J = np.float64(0)

        q = np.sqrt(k**2 - kz**2)

        out = J * self._RadialFunction(q) * self._AxialFunction(kz)

        kz = np.abs(k) * t
        J = np.abs(k)
        q = np.sqrt(k**2 - kz**2)

        out += J * self._RadialFunction(q) * self._AxialFunction(kz)

        return out

    def _1d_integrands(self):
        return {'separated': lambda k, t:
                self._1d_integrand_separated(k, t),
                'enveloped': lambda k, t:
                self._1d_integrand_enveloped(k, t)}

    def _1dComplexIntegration(self,
                              k: np.complex128):

        lower = np.array([0.], np.float64)
        upper = np.array([1.], np.float64)

        def REintegrand(t): return self._1d_integrand(k, t).real
        def IMintegrand(t): return self._1d_integrand(k, t).imag

        RE = np.array(cubature(REintegrand, 1, 1,
                               lower, upper,
                               relerr=self.relerr)).reshape(2,)
        IM = np.array(cubature(IMintegrand, 1, 1,
                               lower, upper,
                               relerr=self.relerr)).reshape(2,)
        res = np.array([RE, IM])

        value = np.sqrt(np.sum(res[:, 0]**2))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                abserr = np.sqrt(
                    np.sum((res[:, 0] * res[:, 1])**2)) / value
            except Warning:
                abserr = np.float64(0)

        return ValueWithError(value=value,
                              abserr=abserr)

    def _6d_integrand_separated_ALP_TM010(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp - self.L - self.d)**2)
        return (rho * self.J(self.prho * self.R * rho) *
                rhop * self.J(self.prho * self.R * rhop) *
                np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r))

    def _6d_integrand_separated_DP_TM010(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp - self.L - self.d)**2)
        return (self.R * rho * self.J(self.prho * self.R * rho) *
                self.R * rhop * self.J(self.prho * self.R * rhop) *
                np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r))

    def _4d_integrand_separated_DP_TM010(self, x, k):
        rho, rhop, phi, phip = x
        r_0 = np.sqrt(self.R**2 * (rho**2 +
                                   rhop**2 -
                                   2 * rho * rhop * np.cos(phi - phip)) +
                      (self.L + self.d)**2)
        r_plus = np.sqrt(self.R**2 * (rho**2 +
                                      rhop**2 -
                                      2 * rho * rhop * np.cos(phi - phip)) +
                         (2 * self.L + self.d)**2)
        r_minus = np.sqrt(self.R**2 * (rho**2 +
                                       rhop**2 -
                                       2 * rho * rhop * np.cos(phi - phip)) +
                          (self.d)**2)
        return (self.R * rho * self.J(self.prho * self.R * rho) *
                self.R * rhop * self.J(self.prho * self.R * rhop) *
                (np.exp(1j * k * r_plus - 1j * k * self.d) /
                 (4 * np.pi * r_plus) -
                 2 * np.exp(1j * k * r_0 - 1j * k * self.d)
                 / (4 * np.pi * r_0) +
                 np.exp(1j * k * r_minus - 1j * k * self.d)
                 / (4 * np.pi * r_minus)))

    def _6d_integrand_separated_TE011(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp - self.L - self.d)**2)
        return 4 * (rho * self.J(self.prho * self.R * rho) *
                    np.sin(self.pz * self.L * z) *
                    rhop * self.J(self.prho * self.R * rhop) *
                    np.sin(self.pz * self.L * zp) *
                    np.cos(phi - phip) *
                    np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r))

    def _6d_integrand_enveloped_DP_TM010(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp)**2)
        m = np.sqrt(self.omega**2 - k**2)
        return (self.R * rho * self.J(self.prho * self.R * rho) *
                self.R * rhop * self.Z(self.prho * self.R * rhop) *
                np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r) *
                (m**2 - 1j * k / r + 1 / r**2 +
                (self.L * z - self.L * zp)**2 *
                (k**2 + 3 * 1j * k / r - 3 / r**2) / r**2))

    def _6d_integrand_enveloped_ALP_TM010(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp)**2)
        return (rho * self.J(self.prho * self.R * rho) *
                rhop * self.Z(self.prho * self.R * rhop) *
                np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r))

    def _6d_integrand_enveloped_TE011(self, x, k):
        rho, rhop, phi, phip, z, zp = x
        r = np.sqrt(self.R**2 * (rho**2 +
                                 rhop**2 -
                                 2 * rho * rhop * np.cos(phi - phip)) +
                    (self.L * z - self.L * zp)**2)
        return 4 * (rho * self.J(self.prho * self.R * rho) *
                    np.sin(self.pz * self.L * z) *
                    rhop * self.Z(self.prho * self.R * rhop) *
                    np.sin(self.pz * self.L * zp) *
                    np.cos(phi - phip) *
                    np.exp(1j * k * r - 1j * k * self.d) / (4 * np.pi * r))

    def _6d_integrands(self):
        return {'separated':
                {'TM010':
                 {'DP': lambda x, k:
                  self._6d_integrand_separated_DP_TM010(x, k),
                  'ALP': lambda x, k:
                  self._6d_integrand_separated_ALP_TM010(x, k)},
                 'TE011':
                 {'DP': lambda x, k:
                  self._6d_integrand_separated_TE011(x, k),
                  'ALP': lambda x, k:
                  self._6d_integrand_separated_TE011(x, k)},
                 },

                'enveloped':
                {'TM010':
                 {'DP': lambda x, k:
                  self._6d_integrand_enveloped_DP_TM010(x, k),
                  'ALP': lambda x, k:
                  self._6d_integrand_enveloped_ALP_TM010(x, k)},
                 'TE011':
                 {'DP': lambda x, k:
                  self._6d_integrand_enveloped_TE011(x, k),
                  'ALP': lambda x, k:
                  self._6d_integrand_enveloped_TE011(x, k)},
                 },
                }

    def _6dComplexIntegration(self,
                              k: np.complex128):

        def REintegrand(x, k): return self._6d_integrand(x, k).real
        def IMintegrand(x, k): return self._6d_integrand(x, k).imag

        RE = monte_carlo(
            6,
            REintegrand,
            k,
            self.lower6d,
            self.upper6d,
            self.calls)
        IM = monte_carlo(
            6,
            IMintegrand,
            k,
            self.lower6d,
            self.upper6d,
            self.calls)

        res = np.array([RE, IM])

        value = np.sqrt(np.sum(res[:, 0]**2))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                abserr = np.sqrt(
                    np.sum((res[:, 0] * res[:, 1])**2)) / value
            except Warning:
                abserr = np.float64(0)

        return ValueWithError(value=value,
                              abserr=abserr)

    def _4d_6d_ComplexIntegration_enveloped_TM010_DP(self,
                                                     k: np.complex128):

        def REintegrand(
            x, k): return self._6d_integrand_separated_DP_TM010(
            x, k).real

        def IMintegrand(
            x, k): return self._6d_integrand_separated_DP_TM010(
            x, k).imag

        RE = monte_carlo(
            6,
            REintegrand,
            k,
            self.lower6d,
            self.upper6d,
            self.calls)
        IM = monte_carlo(
            6,
            IMintegrand,
            k,
            self.lower6d,
            self.upper6d,
            self.calls)

        res6d = np.array([RE, IM])

        def REintegrand(
            x, k): return self._4d_integrand_separated_DP_TM010(
            x, k).real

        def IMintegrand(
            x, k): return self._4d_integrand_separated_DP_TM010(
            x, k).imag

        RE = monte_carlo(
            4,
            REintegrand,
            k,
            self.lower4d,
            self.upper4d,
            self.calls)
        IM = monte_carlo(
            4,
            IMintegrand,
            k,
            self.lower4d,
            self.upper4d,
            self.calls)

        res4d = np.array([RE, IM])

        m = np.sqrt(self.omega**2 - k**2)
        res = m**2 * res6d - res4d / self.L**2

        value = np.sqrt(np.sum(res[:, 0]**2))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                abserr = np.sqrt(
                    np.sum((res[:, 0] * res[:, 1])**2)) / value
            except Warning:
                abserr = np.float64(0)

        return ValueWithError(value=np.abs(value),
                              abserr=np.abs(abserr))

    def _RightAsymptotic_separated_TM010_DP(self):

        theta = np.vectorize(lambda x: np.float64(1.) if x > 0 else np.nan)
        eta = np.vectorize(lambda x: np.float64(x) if x > 0 else np.nan)

        out = np.abs(theta(self.masses.array.value - self.omega) *
                     (self.prho * self.R * self.Jp(self.prho * self.R))**2 *
                     1 / (np.pi * self.R**5 * self.L**2) *
                     1 / (self.masses.array.value**4))

        def _integral_K0(a):
            if a < 1e-10:
                res = np.float64(0)
            elif a < 10:
                res = (special.kv(0, a) * special.modstruve(-1, a) +
                       special.kv(1, a) * special.modstruve(0, a))
            else:
                res = 1 / a - np.sqrt(2 / np.pi) * np.exp(-a) / a**(3 / 2)       
            return  0.5 * np.pi * a**2 * res       
        
        _integral_K0 = np.vectorize(_integral_K0)

        def _integral_K0_asympt1(a):
            if a < 1e-10:
                return np.float64(0)
            else:
                return - np.sqrt(np.pi * a / 2) * np.exp(-a)

        _integral_K0_asympt1 = np.vectorize(_integral_K0_asympt1)

        def _integral_K0_asympt2(a):
            if a > 10:
                return np.sqrt(2 * np.pi * a) * np.exp(-a)
            else:
                return a**2 * (special.kn(2, a) - special.kn(0, a))

        _integral_K0_asympt2 = np.vectorize(_integral_K0_asympt2)

        array_left = self.masses.array.value * self.d < np.float64(10)
        array_right = self.masses.array.value * self.d >= np.float64(10)

        integral = ((_integral_K0(self.masses.array.value * self.d) +
                     _integral_K0(self.masses.array.value * (self.d + 2 * self.L)) -
                 2 * _integral_K0(self.masses.array.value * (self.d + self.L))) *
                    array_left)

        integral += ((_integral_K0_asympt1(self.masses.array.value * self.d) +
                      _integral_K0_asympt1(self.masses.array.value * (self.d + 2 * self.L)) -
                 2 *  _integral_K0_asympt1(self.masses.array.value * (self.d + self.L))) *
                    array_right)
        
        integral += (_integral_K0_asympt2(self.masses.array.value * self.d) +
                     _integral_K0_asympt2(self.masses.array.value * (self.d + 2 * self.L)) -
                  2 *_integral_K0_asympt2(self.masses.array.value * (self.d + self.L)))

        return np.abs(out * integral)

    def _RightAsymptotic_separated_TM010_ALP(self):

        theta = np.vectorize(lambda x: np.float64(1.) if x > 0 else np.nan)
        eta = np.vectorize(lambda x: np.float64(x) if x > 0 else np.nan)

        return np.abs(theta(self.masses.array.value - self.omega) *
                      (self.prho * self.R *
                       self.Jp(self.prho * self.R))**2 *
                      self.Y(self.prho * self.R) *
                      self.Jp(self.prho * self.R) *
                      1 / (4 * self.prho * self.R**3) * np.pi *
                      1 / self.L**2 *
                      np.exp(-self.d * self.masses.array.value) /
                      self.masses.array.value**3)

    def _RightAsymptotic_separated_TE011(self):

        theta = np.vectorize(lambda x: np.float64(1.) if x > 0 else np.nan)
        eta = np.vectorize(lambda x: np.float64(x) if x > 0 else np.nan)

        mu = np.sqrt(eta(self.masses.array.value**2 - self.pz**2))

        return np.abs(theta(self.masses.array.value - self.omega) *
                      (self.prho * self.R *
                       self.Jp(self.prho * self.R))**2 *
                      self.Y(self.prho * self.R) *
                      self.Jp(self.prho * self.R) *
                      1 / (self.prho * self.R**3) *
                      np.pi**3 * 1 / self.L**4 *
                      np.exp(-self.d * mu) /
                      (self.masses.array.value**4 * mu))

    def _RightAsymptotic_enveloped(self):

        theta = np.vectorize(lambda x: np.float64(1.) if x > 0 else np.nan)
        eta = np.vectorize(lambda x: np.float64(x) if x > 0 else np.nan)

        kappa = np.sqrt(eta(self.masses.array.value**2 - self.omega**2))

        return np.abs(theta(self.masses.array.value - self.omega) *
                      self.prho * self.R1 * self.Jp(self.prho * self.R1) *
                      self.prho * self.R2 * self.Zp(self.prho * self.R2) *
                      1 / (self.R1**4 * np.sqrt(self.R1 * self.R2)) *
                      1 / self.L * np.exp(-self.d * kappa) / (kappa**5))

    def _AsymptoticFormFactors(self):

        return {
            'separated': {
                'TM010': {
                    'DP': lambda:
                        self.R**2 *
                        self._RightAsymptotic_separated_TM010_DP(),
                    'ALP': lambda:
                        self._RightAsymptotic_separated_TM010_ALP()},
                'TE011': {
                    'DP': lambda:
                        self._RightAsymptotic_separated_TE011(),
                    'ALP': lambda:
                        self._RightAsymptotic_separated_TE011()},
            },
            'enveloped': {
                'TM010': {
                    'DP': lambda:
                        (self.masses.array.value *
                         self.R1)**2 *
                        self._RightAsymptotic_enveloped(),
                    'ALP': lambda:
                        self._RightAsymptotic_enveloped()},
                'TE011': {
                    'DP': lambda: 2 *
                    self._RightAsymptotic_enveloped(),
                    'ALP': lambda: 2 *
                    self._RightAsymptotic_enveloped(),
                }}}

    def _CalculateFormFactor(self) -> None:

        k = np.sqrt(np.complex128(self.omega**2 - self.masses.array.value**2))
        out = {'1d-integration': [],
               '6d-integration': []}
        print('1d-integration... ', end='')
        for _k in tqdm(k):
            out['1d-integration'].append(self._1dComplexIntegration(_k))

        self.FormFactor['1d-integration'] *= ValueWithError_from_array(
            out['1d-integration'])

        self.FormFactor['right_asymptotic'] *= self._AsymptoticFormFactor()

        if self.calculate_6d:
            print('6d-integration... ', end='')
            if (self.setup_geometry == 'separated' and
                self.cavity_mode == 'TM010' and
                    self.particles_type == 'DP'):
                for _k in tqdm(k[::self.step_of_6d_points]):
                    out['6d-integration'].append(
                        self._4d_6d_ComplexIntegration_enveloped_TM010_DP(_k))
            else:
                for _k in tqdm(k[::self.step_of_6d_points]):
                    out['6d-integration'].append(
                        self._6dComplexIntegration(_k))

            self.FormFactor['6d-integration'] *= ValueWithError_from_array(
                out['6d-integration'])
            self.FormFactor['6d-integration'] *= np.abs(np.exp(
                1j * k[::self.step_of_6d_points] * self.d))

    def _CalculateSensitivity(self) -> None:

        omega = DimensionalVariable(
            array=self.omega,
            dimension='meter',
            degree=-1,)
        V1 = DimensionalVariable(
            array=self.V1,
            dimension='meter',
            degree=3,)
        V2 = DimensionalVariable(
            array=self.V2,
            dimension='meter',
            degree=3,)

        match self.particles_type:

            case 'ALP':

                out = (2 *
                       temperature *
                       Signal_to_Noise_Ratio *
                       omega**(-3) *
                       quality_factor_NRF**(-1) *
                       pump_mode_amplitude_NRF**(-2) *
                       external_magnetic_field**(-4) *
                       V1**(-2) * V2**(-1) *
                       time**(-1))**(1 / 4)

                out1d = out * np.ones(self.len)
                out6d = out * np.ones(self.number_of_6d_points)

            case 'DP':

                out = (2 *
                       temperature *
                       Signal_to_Noise_Ratio *
                       omega**(-3) *
                       quality_factor_SRF**(-1) *
                       pump_mode_amplitude_SRF**(-2) *
                       V1**(-2) * V2**(-1) *
                       time**(-1))**(1 / 4)

                out1d = out * self.masses**(-1)
                out6d = out * DimensionalVariable(
                    self.masses.array.value[::self.step_of_6d_points]**(-1),
                    dimension='meter',
                    degree=1)

        self.Sensitivity = {'1d-integration': out1d,
                            '6d-integration': out6d,
                            'right_asymptotic': out1d}

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for method in ['1d-integration', 'right_asymptotic']:
                self.Sensitivity[method] *= self.FormFactor[method]**(-1 / 2)

        if self.calculate_6d:
            self.Sensitivity['6d-integration'] *= (
                self.FormFactor['6d-integration']**(-1 / 2))

    def Calculate(self,
                  calculate_6d=True) -> None:

        self.calculate_6d = calculate_6d

        self._CalculateFormFactor()
        self._CalculateSensitivity()

        if self._show_image:

            fig = plt.figure(figsize=(7, 5))
            ax = plt.subplot()

            match self.particles_type:

                case 'DP':
                    ax.set_xlabel(r"$m_{A'}\, [{\rm eV}]$")
                    ax.set_ylabel(r'$\epsilon$')

                case 'ALP':
                    ax.set_xlabel(r"$m_{a}\, [{\rm eV}]$")
                    ax.set_ylabel(r'$g_{a\gamma\gamma} \, [{\rm GeV}^{-1}]$')
            colors = {'1d-integration': 'blue',
                  'right_asymptotic': 'red'}
            for method in ('1d-integration', 'right_asymptotic'):
                x = self.masses.eV.value
                y = self.Sensitivity[method].GeV.value
                ax.loglog(x, y, linewidth=3, color=colors[method])

            ax.grid(True,
                which="major",
                ls='solid',
                linewidth=0.5,
                color='0')
            ax.grid(True,
                which="minor",
                ls='dashed',
                linewidth=0.2,
                color='0')

            if self.calculate_6d:
                x = self.masses.eV.value[::self.step_of_6d_points]
                y = self.Sensitivity['6d-integration'].GeV.value
                err = self.Sensitivity['6d-integration'].GeV.abserr
                ax.errorbar(x, y, err, fmt='o', color='green')

            ax.legend(
                labels=[
                    '1d-integration',
                    'right-asymptotic',
                    '6d-integration'],
                loc='best',
                shadow=True,
                framealpha=1,
                facecolor='whitesmoke')

            x = self.masses.eV.value
            y = self.Sensitivity['1d-integration'].GeV.value
            err = self.Sensitivity['1d-integration'].GeV.abserr
            ax.fill_between(x, y - err, y + err, alpha=0.5, color='blue')

            plt.show()

    def _CalculateAnalyticalRightAsymptotic(self):

        max_mass = self.masses.array.value[-1]
        masses = DimensionalVariable(
                array=np.logspace(np.log(self.min_mass) / np.log(10),
                              np.log(max_mass) / np.log(10), 100),
            dimension='meter',
            degree=-1)

        omega = DimensionalVariable(
            array=self.omega,
            dimension='meter',
            degree=-1,)
        R = DimensionalVariable(
            array=self.R,
            dimension='meter',
            degree=1,)
        L = DimensionalVariable(
            array=self.L,
            dimension='meter',
            degree=1,)

        kappa = (masses**2 - omega**2)**(1 / 2)

        if self.cavity_mode == 'TE011':
            pz = DimensionalVariable(
                array=self.pz,
                dimension='meter',
                degree=-1,)
            mu = (masses**2 - pz**2)**(1 / 2)

        match self.particles_type:

            case 'ALP':

                out = (2 *
                       temperature *
                       Signal_to_Noise_Ratio *
                       quality_factor_NRF**(-1) *
                       pump_mode_amplitude_NRF**(-2) *
                       external_magnetic_field**(-4) *
                       time**(-1))**(1 / 4)

                out *= np.ones(len(masses.array.value))

            case 'DP':

                out = (2 *
                       temperature *
                       Signal_to_Noise_Ratio *
                       quality_factor_SRF**(-1) *
                       pump_mode_amplitude_SRF**(-2) *
                       time**(-1))**(1 / 4)

                out *= np.ones(len(masses.array.value))

        res = {'separated':
               {'TM010':
                {'DP': lambda: {r"$m_{A'}d \ll 1$" : (
                                    (np.pi / self.x)**(1 / 4) *
                                    (L * R)**(1 / 4) *
                                    masses / 2**(1 / 2)),
                                r"$m_{A'}d \gg 1$" : (
                                    (np.pi / self.x)**(1 / 4) *
                                    (L * R)**(1 / 4) *
                                    masses * 
                                    (2 / np.pi)**(1 / 4) /
                                    (masses.array.value *
                                     self.d)**(1 / 4) *
                                    np.exp(masses.array.value *
                                           self.d / 2)
                                    )},
                 'ALP': lambda: {r"precise asymptotic" : (2 *
                                 (1 / np.pi**3 / self.x**3 /
                                  (self.Y(self.x) *
                                   self.Jp(self.x))**2)**(1 / 4) *
                                 (self.L / self.R)**(1 / 4) *
                                 masses**(3 / 2) *
                                 np.exp(masses.array.value * self.d / 2)
                                 )}},
                'TE011':
                {'DP': lambda: {r"precise asymptotic" : 
                                (1 / np.pi**(1 / 2) *
                                (self.omega**3 * self.L**5 /
                                 np.pi**3 / self.R**2)**(1 / 4) *
                                masses**(1 / 2) *
                                np.exp(masses.array.value * self.d / 2)),
                                r"$R \gg L$" : (1 / np.pi**(1 / 2) *
                                (self.L / self.R)**(1 / 2) *
                                masses**(1 / 2) *
                                np.exp(masses.array.value * self.d / 2))},
                 'ALP': lambda: {r"precise asymptotic" :
                                 (2 / np.pi**(1 / 2) *
                                 (L**5 / np.pi**3 / omega / R**2)**(1 / 4) *
                                 masses**2 * mu**(1 / 2) *
                                 np.exp(mu.array.value * self.d / 2)),
                                 r"$R \gg L$" : (2 / np.pi**2 *
                                 (1 / self.x /
                                  np.abs(self.Y(self.x) *
                                         self.Jp(self.x)))**(1 / 2) *
                                 (L**3 / R)**(1 / 2) *
                                 masses**2 * mu**(1 / 2) *
                                 np.exp(mu.array.value * self.d / 2))}
                 }
                },
               'enveloped':
               {'TM010':
                {'DP': lambda: {r"precise asymptotic" : 
                                ((self.R2 / self.R1 *
                                 np.abs((self.R3 *
                                         self.Zp(self.x * self.R3 / self.R1) /
                                         self.R2 /
                                         self.Zp(self.x * self.R2 / self.R1)
                                         )**2 - 1) /
                                (np.pi * self.x))**(1 / 4) *
                                (self.R1 / self.L)**(1 / 4) *
                                kappa**(5 / 2) / masses**2 *
                                np.exp(kappa.array.value * self.d / 2)),
                                r"Approximation $d \ll R$ and $m_{A'} \gg \omega$" : ((
                                 np.abs((self.zeros(2) * self.Jp(self.zeros(2)) /
                                         self.zeros(1) / self.Jp(self.zeros(1)))**2 
                                        - 1) / (np.pi * self.x))**(1 / 4) *
                                (self.R1 / self.L)**(1 / 4) *
                                masses**(1 / 2) *
                                np.exp(masses.array.value * self.d / 2)),
                                },
                 'ALP': lambda: {r"precise asymptotic" : ((((self.R3 / self.R1)**2 -
                                   (self.R2 / self.R1)**2) *
                                  np.abs((self.R3 *
                                          self.Zp(self.x * self.R3 / self.R1) /
                                          self.R2 /
                                          self.Zp(self.x * self.R2 / self.R1)
                                          )**2 - 1) /
                                  (np.pi * self.x**5 * self.R2 / self.R1) /
                                  ((self.R3 / self.R2)**2 - 1))**(1 / 4) *
                                 (R**5 / L)**(1 / 4) *
                                 kappa**(5 / 2) *
                                 np.exp(kappa.array.value * self.d / 2)),
                                 r"Approximation $d \ll R$": ((
                                  np.abs((self.zeros(2) * self.Jp(self.zeros(2)) /
                                          self.zeros(1) / self.Jp(self.zeros(1))
                                          )**2 - 1) /
                                  (np.pi * self.x**5))**(1 / 4) *
                                  (R**5 / L)**(1 / 4) *
                                 kappa**(5 / 2) *
                                 np.exp(kappa.array.value * self.d / 2))
                                 }
                 },
                'TE011':
                {'DP': lambda: {r"precise asymptotic" : 
                                ((self.R2 / self.R1 *
                                 np.abs((self.R3 *
                                         self.Zp(self.x * self.R3 / self.R1) /
                                         self.R2 /
                                         self.Zp(self.x * self.R2 / self.R1)
                                         )**2 - 1) /
                                 (np.pi * self.x**4))**(1 / 4) *
                                (self.omega**3 * self.R1**4 /
                                 self.L)**(1 / 4) *
                                kappa**(5 / 2) / masses**2 *
                                np.exp(kappa.array.value * self.d / 2)),
                                r"$L \gg R_1$" : 
                                ((np.abs((self.zeros(2) *
                                          self.Jp(self.zeros(2)) /
                                          self.zeros(1) /
                                          self.Jp(self.zeros(1))
                                          )**2 - 1) /
                                (np.pi * self.x))**(1 / 4) *
                                (self.R1 / self.L)**(1 / 4) *
                                masses**(1 / 2) *
                                np.exp(masses.array.value * self.d / 2)), 
                                },
                 'ALP': lambda: {r"precise asymptotic" : ((4 *
                                  ((self.R3 / self.R1)**2 -
                                   (self.R2 / self.R1)**2) *
                                  np.abs((self.R3 *
                                          self.Zp(self.x * self.R3 / self.R1) /
                                          self.R2 /
                                          self.Zp(self.x * self.R2 / self.R1)
                                          )**2 - 1) /
                                  (np.pi * self.x**4 * self.R2 / self.R1) /
                                  ((self.R3 / self.R2)**2 - 1))**(1 / 4) *
                                 (R**4 / omega / L)**(1 / 4) *
                                 kappa**(5 / 2) *
                                 np.exp(kappa.array.value * self.d / 2)), 
                                 r"$L \gg R_1$" : ((4 *
                                  ((self.R3 / self.R1)**2 -
                                   (self.R2 / self.R1)**2) *
                                  np.abs((self.R3 *
                                          self.Zp(self.x * self.R3 / self.R1) /
                                          self.R2 /
                                          self.Zp(self.x * self.R2 / self.R1)
                                          )**2 - 1) /
                                  (np.pi * self.x**5 * self.R2 / self.R1) /
                                  ((self.R3 / self.R2)**2 - 1))**(1 / 4) *
                                 (R**5 / L)**(1 / 4) *
                                 kappa**(5 / 2) *
                                 np.exp(kappa.array.value * self.d / 2)),
                                 }}}}

        formulas = {'separated':
                    {'TM010':
                     {'DP': (r"${\tt Asymptotic\ for}\ m_{A'}R \gg 1: \\" +
                             r"{\tt 1.\ For}\ m_{A'}d \ll 1: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}}" +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{\pi}{x_{01}}\right]^{1/4}\times " +
                             r"(R\cdot L)^{1/4} \times" +
                             r"2^{-1/2} \times m_{A'}; \\" +
                             r"{\tt 2.\ For}\ m_{A'}d \gg 1: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}}" +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{\pi}{x_{01}}\right]^{1/4}\times " +
                             r"(R\cdot L)^{1/4} \times " +
                             r"m_{A'} \times \left(\dfrac{2}{\pi}\right)^{1/4} \times" + 
                             r"\dfrac{\exp\left(\frac{m_{A'}d}{2}\right)} " + 
                             r"{(m_{A'}d)^{1/4}}.$"),
                      'ALP': (r"${\tt Asymptotic\ for}\ m_{A'}L \gg 1: \\" +
                              r"g_{a\gamma\gamma} = " +
                              r"\left[\dfrac{2 T \, {\rm SNR}}{Q_{\rm rec} " +
                              r"E_0^2 B_{\rm ext}^4 t}\right]^{1/4} \times " +
                              r"2\cdot \left[\dfrac{1} " +
                              r"{\pi^3 x^3_{01}" +
                              r"(Y_0(x_{01})J'_0(x_{01})^2)}" +
                              r"\right]^{1/4}\times " +
                              r"\left(\dfrac{L}{R}\right)^{1/4} \times " +
                              r"m^{3/2}_a \times " +
                              r"\exp\left(\dfrac{m_a d}{2}\right) $"
                              )},
                        'TE011':
                        {'DP': (r"${\tt Asymptotic\ for}\ m_{A'}L \gg 1: \\" +
                                r"{\tt 1.\ Precise\ asymptotic}: \\" +
                                r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}}" +
                                r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                                r"\dfrac{1}{\pi^{1/2}}" +
                                r"\times \left(\dfrac{\omega^3L^5} " +
                                r"{\pi^3R^2}\right)^{1/4}" +
                                r"\times m_{A'}^{1/2} " +
                                r"\times \exp\left(\dfrac{m_{A'}d}{2}\right) \\"
                                r"{\tt 2.\ Approximation}\ R \gg L: \\" +
                                r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}}" +
                                r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                                r"\dfrac{1}{\pi^{1/2}}" +
                                r"\times \left(\dfrac{L}" +
                                r"{R}\right)^{1/2}" +
                                r"\times m_{A'}^{1/2} " +
                                r"\times \exp\left(\dfrac{m_{A'}d}{2}\right)$"
                                ),
                         'ALP': (r"${\tt Asymptotic\ for}\ m_aL \gg 1: \\" +
                                 r"{\tt 1.\ Precise\ asymptotic}: \\" +
                                 r"g_{a\gamma\gamma} = " +
                                 r"\left[\dfrac{2 T \, {\rm SNR}} " +
                                 r"{Q_{\rm rec} E_0^2 B_{\rm ext}^4 t}" +
                                 r"\right]^{1/4} \times" +
                                 r"\dfrac{2}{\pi} \left[\dfrac{1} " +
                                 r"{x_{11}|Y_1(x_{11})J'_1(x_{11})|}" +
                                 r"\right]^{1/2} \times" +
                                 r"\left(\dfrac{L^5}{\pi^3 \omega R^2}" +
                                 r"\right)^{1/4}" +
                                 r"\times m_a^2 \cdot \mu^{1/2} \times " +
                                 r"\exp\left(\dfrac{m_{A'}d}{2}\right) \\" +
                                 r"{\tt 2.\ Approximation}\ R \gg L: \\" +
                                 r"g_{a\gamma\gamma} = " +
                                 r"\left[\dfrac{2 T \, {\rm SNR}} " +
                                 r"{Q_{\rm rec} E_0^2 B_{\rm ext}^4 t}" +
                                 r"\right]^{1/4} \times" +
                                 r"\dfrac{2}{\pi^2} \left[\dfrac{1} " +
                                 r"{x_{11}|Y_1(x_{11})J'_1(x_{11})|}" +
                                 r"\right]^{1/2} \times" +
                                 r"\left(\dfrac{L^3}{R}" +
                                 r"\right)^{1/2}" +
                                 r"\times m_a^2 \cdot \mu^{1/2} \times " +
                                 r"\exp\left(\dfrac{m_{A'}d}{2}\right)$"
                                )
                         }
                        },
                    'enveloped':
                    {'TM010':
                     {'DP': (r"${\tt Asymptotic\ for}\ " +
                             r"m_{A'}R_1 \gg 1: \\" +
                             r"{\tt 1.\ Precise\ asymptotic}: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}} " +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{R_2}{R_1}\times \dfrac{" +
                             r"\left(\frac{R_3 Z'_0(p_\rho R_3)}" + 
                             r"{R_2 Z'_0(p_\rho R_2)}\right)^2" +
                             r"- 1}{\pi x_{01}}" + 
                             r"\right]^{1/4} \times" +
                             r"\left(\dfrac{R_1}{L}\right)^{1/4} \times " +
                             r"\varkappa_{A'}^{1/2}" +
                             r"\left(\dfrac{\varkappa_{A'}} " +
                             r"{m_{A'}} \right)^2 " + 
                             r"\times \exp\left(\dfrac{m_{A'}d}" +
                             r"{2}\right) \\" +
                             r"{\tt 2.\ Approximation}\ d \ll R_1: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}} " +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{" +
                             r"\left(\frac{x_{02}J'_0(x_{02})}" + 
                             r"{x_{01}J'_0(x_{01})}\right)^2" +
                             r"- 1}{\pi x_{01}}" + 
                             r"\right]^{1/4} \times" +
                             r"\left(\dfrac{R_1}{L}\right)^{1/4} \times " +
                             r"m_{A'}^{1/2} \cdot " +
                             r"\times \exp\left(\dfrac{m_{A'}d}" +
                             r"{2}\right)$"),
                        'ALP': (r"${\tt Asymptotic\ for}\ " +
                                 r"m_{A'}R_1 \gg 1: \\" +
                                 r"{\tt 1.\ Precise\ asymptotic}:\\"
                                 r"g_{a\gamma\gamma} = \left[\dfrac{2 T \, " +
                                 r"{\rm SNR}}{Q_{\rm rec} " +
                                 r" E_0^2 B_{\rm ext}^4 t}" +
                                 r"\right]^{1/4} \times" +
                                 r"\left[\dfrac{\left[(\frac{R_3}{R_1})^2 - " +
                                 r"(\frac{R_2}{R_1})^2\right]}{\pi x^5_{01} " +
                                 r"\cdot \frac{R_2}{R_1}}\times " +
                                 r"\dfrac{\left[\frac{R_3}{R_2} \times " +
                                 r"\frac{ Z'_0(x_{01} \frac{R_3}{R_1})}" +
                                 r"{Z'_0(x_{01} \frac{R_2}" +
                                 r"{R_1})}\right]^2 - 1}" +
                                 r"{(\frac{R_3}{R_2})^2 - " +
                                 r"1}\right]^{1/4} \times" +
                                 r"\left(\dfrac{R_1^5}{L}\right)^{1/4} " +
                                 r"\times \varkappa^{5/2} \times " +
                                 r"\exp\left(\dfrac{\varkappa d}{2}\right);\\" + 
                                 r"{\tt 2.\ Approximation}\ d \ll R_1:\\" +
                                 r"g_{a\gamma\gamma} = \left[\dfrac{2 T \, " +
                                 r"{\rm SNR}}{Q_{\rm rec} " +
                                 r" E_0^2 B_{\rm ext}^4 t}" +
                                 r"\right]^{1/4} \times" +
                                 r"\left[\dfrac{\left(" + 
                                 r"\frac{x_{02}J'_0(x_{02})}" +
                                 r"{x_{01}J'_0(x_{01})} \right)^2 - 1}" +
                                 r"{\pi x^5_{01}}\right]^{1/4} \times" +
                                 r"\left(\dfrac{R_1^5}{L}\right)^{1/4} " +
                                 r"\times \varkappa^{5/2} \times " +
                                 r"\exp\left(\dfrac{\varkappa d}{2}\right)$.")},
                        'TE011':
                        {'DP': (r"${\tt Asymptotic\ for}\ " +
                             r"m_{A'}R_1 \gg 1: \\" +
                             r"{\tt 1.\ Precise\ asymptotic}: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}} " +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{R_2}{R_1}\times \dfrac{" +
                             r"\left(\frac{R_3 Z'_1(p_\rho R_3)}" + 
                             r"{R_2 Z'_1(p_\rho R_2)}\right)^2" +
                             r"- 1}{\pi x^4_{11}}" + 
                             r"\right]^{1/4} \times" +
                             r"\left(\dfrac{\omega^3 R^4_1}{L}\right)^{1/4} \times " +
                             r"\varkappa_{A'}^{1/2}" +
                             r"\left(\dfrac{\varkappa_{A'}} " +
                             r"{m_{A'}} \right)^2 " + 
                             r"\times \exp\left(\dfrac{m_{A'}d}" +
                             r"{2}\right) \\" +
                             r"{\tt 2.\ Approximation}\ d \ll R_1: \\" +
                             r"\epsilon = \left[\dfrac{2 T \, {\rm SNR}} " +
                             r"{Q_{\rm rec} E_0^2 t}\right]^{1/4} \times" +
                             r"\left[\dfrac{" +
                             r"\left(\frac{x_{12}J'_1(x_{12})}" + 
                             r"{x_{11}J'_1(x_{11})}\right)^2" +
                             r"- 1}{\pi x_{11}}" + 
                             r"\right]^{1/4} \times" +
                             r"\left(\dfrac{R_1}{L}\right)^{1/4} \times " +
                             r"m_{A'}^{1/2} \cdot " +
                             r"\times \exp\left(\dfrac{m_{A'}d}" +
                             r"{2}\right)$"),
                         'ALP': (r"${\tt Asymptotic\ for}\ " +
                                 r"m_{A'}R_1 \gg 1: \\" +
                                 r"g_{a\gamma\gamma} = " +
                                 r"\left[\dfrac{2 T \, {\rm SNR}}" +
                                 r"{Q_{\rm rec} E_0^2 B_{\rm ext}^4 t}" +
                                 r"\right]^{1/4} \times" +
                                 r"\left[\dfrac{4 \times " +
                                 r"\left[(\frac{R_3}{R_1})^2 - " +
                                 r"(\frac{R_2}{R_1})^2\right]}{\pi x^4_{01}" +
                                 r"\cdot \frac{R_2}{R_1}}\times " +
                                 r"\dfrac{\left[\frac{R_3}{R_2} \times " +
                                 r"\frac{ Z'_1(x_{11} \frac{R_3}{R_1})} " +
                                 r"{Z'_1(x_{11} \frac{R_2}{R_1})}\right]^2 " +
                                 r"- 1}{(\frac{R_3}{R_2})^2" +
                                 r"- 1}\right]^{1/4} " +
                                 r"\times \left(\dfrac{R_1^4}" +
                                 r"{\omega L}\right)^{1/4} \times " +
                                 r"\varkappa^{5/2} \times " +
                                 r"\exp\left(\dfrac{\varkappa d}" +
                                 r"{2}\right)$")}}}

        if self._show_image:
            display(Latex(formulas[self.setup_geometry]
                    [self.cavity_mode][self.particles_type]))

        _dict = res[self.setup_geometry][self.cavity_mode][self.particles_type]()

        self.AnalyticalRightAsymptotic = {'x': masses}
        
        for key, val in _dict.items():
            self.AnalyticalRightAsymptotic[key] = val * out
            
    def get_plot_data(self,
                      error_factor: float = 5.,
                      min_mass_to_omega_ratio = 2,
                      save_fig = None):

        self.min_mass = min_mass_to_omega_ratio * self.omega
        self._CalculateAnalyticalRightAsymptotic()

        smoothed = DimensionalVariable(
            array=np.zeros(
                self.len),
            dimension=self.Sensitivity['1d-integration'].dimension,
            degree=self.Sensitivity['1d-integration'].degree)

        for i in range(self.len):
            if (self.Sensitivity['1d-integration'].array.relerr[i] <
                    error_factor * self.relerr):
                smoothed.array[i] = self.Sensitivity['1d-integration'].array[i]
            else:
                smoothed.array[i] = self.Sensitivity['right_asymptotic'].array[i]

        smoothed.update()
        self.Sensitivity['smoothed'] = smoothed

        if self._show_image:

            fig = plt.figure(figsize=(7, 5))
            ax = plt.subplot()

            match self.particles_type:

                case 'DP':
                    ax.set_xlabel(r"$m_{A'}\, [{\rm eV}]$")
                    ax.set_ylabel(r'$\epsilon$')

                case 'ALP':
                    ax.set_xlabel(r"$m_{a}\, [{\rm eV}]$")
                    ax.set_ylabel(r'$g_{a\gamma\gamma} \, [{\rm GeV}^{-1}]$')

            x = self.masses.eV.value
            y = self.Sensitivity['smoothed'].GeV.value
            ax.loglog(x, y, linewidth=3,
                    color='green')

            colors = ['red', 'blue']
            j = 0

            for key, val in self.AnalyticalRightAsymptotic.items():
                if key != 'x':
                    x = self.AnalyticalRightAsymptotic['x'].eV.value
                    y = val.GeV.value
                    ax.loglog(x, y, linewidth=3, linestyle='--', color=colors[j])
                    j += 1

            ax.grid(True,
                which="major",
                ls='solid',
                linewidth=0.4,
                color='0')
            ax.grid(True,
                which="minor",
                ls='dashed',
                linewidth=0.1,
                color='0')

            ax.legend(
                labels=[
                    'smoothed line'] + list(
                        self.AnalyticalRightAsymptotic.keys())[1:],
                loc='best',
                shadow=True,
                framealpha=1,
                facecolor='whitesmoke')

            ax.set_title((f'{self.setup_geometry} geometry; ' +
                      f'{self.cavity_mode}-mode; ' +
                      f'particles:{self.particles_type}.'))
            if save_fig is not None:
                plt.savefig('figures/' + save_fig, bbox_inches = 'tight', facecolor='white')
            else:
                plt.show()
            print('the plot data were written to the .plot_data attribute')
            print((f'x-array: dimension = eV ^ ' +
                f'{self.masses.natural_dimension:.1f}, ' +
                f'len = {len(self.masses.eV.value)}'))
            print((f'y-array: dimension = eV ^ ' +
                f"{self.Sensitivity['smoothed'].natural_dimension:.1f}, " +
                f"len = {len(self.Sensitivity['smoothed'].GeV.value)}"))
            self.plot_data = {'x': self.masses.eV.value,
                              'y': self.Sensitivity['smoothed'].GeV.value}
