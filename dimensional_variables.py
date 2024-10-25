import numpy as np

REDUCED_PLANCK_CONSTANT = 1.054571e-34
SPEED_OF_LIGHT = 2.997924e+08
ELEMENTARY_CHARGE = 1.602176e-19
BOLTZMANN_CONSTANT = 1.380649e-23


class ValueWithError:
    def __init__(self,
                 value: np.int32 | np.float64 | np.ndarray,
                 abserr: np.int32 | np.float64 | np.ndarray = None,
                 relerr: np.int32 | np.float64 | np.ndarray = None) -> None:

        if type(value) in (int, np.int32, float, np.float64):
            self.value = np.array([np.float64(value)])
        elif type(value) in (list, np.ndarray):
            self.value = np.array(value, np.float64)

        self.shape = self.value.shape
        self.ndim = self.shape[0]

        if abserr is not None:
            self.abserr = np.abs(abserr) * np.ones(self.shape, np.float64)
            self.relerr = np.abs(self.abserr / self.value)
        elif relerr is not None:
            self.relerr = np.abs(relerr) * np.ones(self.shape, np.float64)
            self.abserr = np.abs(self.relerr * self.value)
        else:
            self.abserr = np.zeros(self.shape)
            self.relerr = np.zeros(self.shape)

    def __setitem__(self, key, other):
        (self.value[key],
         self.abserr[key],
         self.relerr[key]) = other

    def __getitem__(self, key: int):
        return (self.value[key],
                self.abserr[key],
                self.relerr[key])

    def __add__(self, other):
        if type(other) in (int, np.int32, float, np.float64,
                           np.ndarray):
            return ValueWithError(
                value=self.value + other,
                abserr=self.abserr,
            )
        if type(other) in ValueWithError:
            return ValueWithError(
                value=self.value + other.value,
                abserr=self.abserr + self.abserr,
            )

    def __sub__(self, other):
        if isinstance(other, (int, np.int32, float, np.float64,
                              np.ndarray)):
            return ValueWithError(
                value=self.value - other,
                abserr=self.abserr,
            )
        if isinstance(other, ValueWithError):
            return ValueWithError(
                value=self.value - other.value,
                abserr=self.abserr + self.abserr,
            )

    def __rsub__(self, other):
        if isinstance(other, (int, np.int32, float, np.float64,
                              np.ndarray)):
            return ValueWithError(
                value=other - self.value,
                abserr=self.abserr,
            )
        if isinstance(other, ValueWithError):
            return ValueWithError(
                value=other.value - self.value,
                abserr=self.abserr + self.abserr,
            )

    def __mul__(self, other):
        if isinstance(other, (int, np.int32, float, np.float64,
                              np.ndarray)):
            return ValueWithError(
                value=self.value * other,
                relerr=self.relerr
            )
        if isinstance(other, ValueWithError):
            return ValueWithError(
                value=self.value * other.value,
                relerr=self.relerr + other.relerr
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, degree):
        return ValueWithError(
            value=self.value**degree,
            relerr=self.relerr * degree,
        )

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return other * self**(-1)

    def reshape(self, *args):
        self.value.reshape(*args)
        self.abserr.reshape(*args)
        self.relerr.reshape(*args)


def ValueWithError_from_array(array):
    return ValueWithError(
        value=np.array([x.value.item() for x in array]),
        abserr=np.array([x.abserr.item() for x in array]),
    )


available_dimensions = [
    None,
    'eV',
    'meter',
    'second',
    'Kelvin',
    'Tesla',
]

convertion_factors = {
    None: 1.,
    'eV': 1.,
    'meter': ELEMENTARY_CHARGE / REDUCED_PLANCK_CONSTANT / SPEED_OF_LIGHT,
    'second': ELEMENTARY_CHARGE / REDUCED_PLANCK_CONSTANT,
    'Kelvin': BOLTZMANN_CONSTANT / ELEMENTARY_CHARGE,
    'Tesla': 1e+4 * 10**(-1 / 2) * SPEED_OF_LIGHT**(3 / 2)
    * ELEMENTARY_CHARGE**(-2) * REDUCED_PLANCK_CONSTANT**(3 / 2)
    * (4 * np.pi)**(-1 / 2),
}

natural_dimensions = {
    None: 0,
    'eV': 1,
    'meter': -1,
    'second': -1,
    'Kelvin': 1,
    'Tesla': 2,
}


class DimensionalVariable:
    def __init__(self,
                 array: (int | np.int32 |
                         float | np.float64 |
                         list | np.ndarray |
                         ValueWithError),
                 dimension: str = None,
                 degree: (int | np.int32 |
                          float | np.float64) = np.float64(1),
                 SI_unit: str = None) -> None:
        if not isinstance(array, ValueWithError):
            self.array = ValueWithError(value=array)
        else:
            self.array = array
        assert dimension in available_dimensions, \
            f'dimension {dimension} is not available'
        self.dimension = dimension
        self.degree = np.float64(degree)
        self.natural_dimension = np.float64(
            natural_dimensions[dimension] * degree)
        self._convert_to_natural_unit()
        self.SI_unit = SI_unit
        if self.SI_unit is not None:
            self._convert_to_SI_unit(SI_unit)

    def _convert_to_natural_unit(self):
        self.eV = (self.array *
                   convertion_factors[self.dimension]**self.degree)
        self.GeV = self.eV * np.power(1e-9, self.natural_dimension)

    def _convert_to_SI_unit(self, dimension):
        assert dimension in available_dimensions, \
            f'dimension {dimension} is not available'
        self.array = (
            self.eV /
            convertion_factors[dimension]**natural_dimensions[dimension])
        self.natural_dimension = self.degree
        self.dimension = dimension
        self.degree = self.natural_dimension / natural_dimensions[dimension]

    def update(self):
        self._convert_to_natural_unit()
        if self.SI_unit is not None:
            self._convert_to_SI_unit(SI_unit)

    def __add__(self, other):
        assert self.dimension == other.dimension \
            and self.degree == other.degree, \
            f'dimensions {self.dimension}**{self.degree}' + \
            'and {other.dimension}**{other.degree} are not the same'
        array = self.array + other.array
        degree = self.degree
        out = DimensionalVariable(
            array=array,
            dimension=self.dimension,
            degree=degree,
        )
        return out

    def __sub__(self, other):
        assert self.dimension == other.dimension \
            and self.degree == other.degree, \
            f'dimensions {self.dimension}**{self.degree}' + \
            'and {other.dimension}**{other.degree} are not the same'
        array = self.array - other.array
        degree = self.degree
        out = DimensionalVariable(
            array=array,
            dimension=self.dimension,
            degree=degree,
        )
        return out

    def __rsub__(self, other):
        assert self.dimension == other.dimension \
            and self.degree == other.degree, \
            f'dimensions {self.dimension}**{self.degree}' + \
            'and {other.dimension}**{other.degree} are not the same'
        array = other.array - self.array
        degree = self.degree
        out = DimensionalVariable(
            array=array,
            dimension=self.dimension,
            degree=degree,
        )
        return out

    def __rmul__(self, other):
        if isinstance(other, (int, float,
                              np.int32, np.float64,
                              np.ndarray, ValueWithError)):
            array = self.array * other
            degree = self.degree
            out = DimensionalVariable(
                array=array,
                dimension=self.dimension,
                degree=degree,
            )
            return out
        else:
            if self.dimension == other.dimension:
                array = self.array * other.array
                degree = self.degree + other.degree
                out = DimensionalVariable(
                    array=array,
                    dimension=self.dimension,
                    degree=degree,
                )
                return out
            else:
                array = self.eV * other.eV
                degree = self.natural_dimension + other.natural_dimension
                out = DimensionalVariable(
                    array=array,
                    dimension='eV',
                    degree=degree,
                )
                return out

    def __mul__(self, other):
        return self.__rmul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float,
                              np.int32, np.float64,
                              np.ndarray)):
            array = self.array / other
            degree = self.degree
            out = DimensionalVariable(
                array=array,
                dimension=self.dimension,
                degree=degree,
            )
            return out
        else:
            if self.dimension == other.dimension:
                array = self.array / other.array
                degree = self.degree - other.degree
                out = DimensionalVariable(
                    array=array,
                    dimension=self.dimension,
                    degree=degree,
                )
                return out
            else:
                array = self.eV / other.eV
                degree = self.natural_dimension - other.natural_dimension
                out = DimensionalVariable(
                    array=array,
                    dimension='eV',
                    degree=degree,
                )
                return out

    def __rtruediv__(self, other):
        if isinstance(other, (int, float,
                              np.int32, np.float64,
                              np.ndarray)):
            array = other / self.array
            degree = self.degree
            out = DimensionalVariable(
                array=array,
                dimension=self.dimension,
                degree=-degree,
            )
            return out
        else:
            if self.dimension == other.dimension:
                array = other.array / self.array
                degree = other.degree - self.degree
                out = DimensionalVariable(
                    array=array,
                    dimension=self.dimension,
                    degree=degree,
                )
                return out
            else:
                array = other.eV / self.eV
                degree = other.natural_dimension - self.natural_dimension
                out = DimensionalVariable(
                    array=array,
                    dimension='eV',
                    degree=degree,
                )
                return out

    def __pow__(self, degree: np.float64):
        array = np.power(self.array, degree)
        degree = self.degree * degree
        out = DimensionalVariable(
            array=array,
            dimension=self.dimension,
            degree=degree,
        )
        return out

    def reshape(self, *args):
        self.array.reshape(*args)
        self.eV.reshape(*args)
        self.GeV.reshape(*args)


def sqrt(x):
    array = np.sqrt(x.array)
    degree = x.degree / 2
    out = DimensionalVariable(
        array=array,
        dimension=x.dimension,
        degree=degree,
    )
    return out


def DimensionalVariable_from_array(array):
    return DimensionalVariable(
        array=ValueWithError(
            value=[x.array.value for x in array],
            abserr=[x.array.abserr for x in array]),
        dimension=array[0].dimension,
        degree=array[0].degree,
    )
