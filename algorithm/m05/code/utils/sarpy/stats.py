import os
import numpy as np
import numpy.typing as npt
import scipy.special as sc
from scipy._lib._util import _lazywhere as lw
from scipy._lib import array_api_compat as aac
from types import ModuleType
from typing import Any
from scipy._lib.array_api_compat import (
    is_array_api_obj as iaao,
    numpy as np_compat,)

SD = os.environ.get("SCIPY_DEVICE", "cpu")
SA: str | bool = os.environ.get("SCIPY_ARRAY_API", False)

GC = {"SCIPY_ARRAY_API": SA, "SCIPY_DEVICE": SD}

Array = Any
ArrayLike = Array | npt.ArrayLike


def cf(arrs: list[ArrayLike]) -> list[Array]:
    for i in range(len(arrs)):
        arr = arrs[i]
        if isinstance(arr, np.ma.MaskedArray):
            raise TypeError("Inputs of type `numpy.ma.MaskedArray` are not supported.")
        elif isinstance(arr, np.matrix):
            raise TypeError("Inputs of type `numpy.matrix` are not supported.")
        if isinstance(arr, (np.ndarray, np.generic)):
            dt = arr.dtype
            if not (np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)):
                raise TypeError(f"An argument has dtype `{dt!r}`; "
                                f"only boolean and numerical dtypes are supported.")
        elif not iaao(arr):
            try:
                arr = np.asanyarray(arr)
            except TypeError:
                raise TypeError("An argument is neither array API compatible nor "
                                "coercible by NumPy.")
            dt = arr.dtype
            if not (np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)):
                msg = (
                    f"An argument was coerced to an unsupported dtype `{dt!r}`; "
                    f"only boolean and numerical dtypes are supported."
                )
                raise TypeError(msg)
            arrs[i] = arr
    return arrs


def an(*arrs: Array) -> ModuleType:
    if not GC["SCIPY_ARRAY_API"]:
        return np_compat

    _arrs = [arr for arr in arrs if arr is not None]

    _arrs = cf(_arrs)

    return aac.array_namespace(*_arrs)


def lw(cond, arrs, f, fillval=None, f2=None):
    xp = an(cond, *arrs)

    if (f2 is fillval is None) or (f2 is not None and fillval is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrs)
    bool_dt = xp.asarray([True]).dtype
    cond, arrs = xp.astype(args[0], bool_dt, copy=False), args[1:]

    tmp1 = xp.asarray(f(*(arr[cond] for arr in arrs)))

    if f2 is None:
        if type(fillval) in {bool, int, float, complex}:
            with np.errstate(invalid='ignore'):
                dt = (tmp1 * fillval).dtype
        else:
            dt = xp.result_type(tmp1.dtype, fillval)
        out = xp.full(cond.shape, dtype=dt,
                      fill_value=xp.asarray(fillval, dtype=dt))
    else:
        ncond = ~cond
        tmp2 = xp.asarray(f2(*(arr[ncond] for arr in arrs)))
        dt = xp.result_type(tmp1, tmp2)
        out = xp.empty(cond.shape, dtype=dt)
        out[ncond] = tmp2

    out[cond] = tmp1

    return out


class _ShapeInfo:
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf),
                 inclusive=(True, True)):
        self.name = name
        self.integrality = integrality

        domain = list(domain)
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain


class fg:
    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds
        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())
        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    def _shape_info(self):
        idfn = _ShapeInfo("dfn", False, (0, np.inf), (False, False))
        idfd = _ShapeInfo("dfd", False, (0, np.inf), (False, False))
        return [idfn, idfd]

    def _rvs(self, dfn, dfd, size=None, random_state=None):
        return random_state.f(dfn, dfd, size)

    def _pdf(self, x, dfn, dfd):
        return np.exp(self._logpdf(x, dfn, dfd))

    def _logpdf(self, x, dfn, dfd):
        n = 1.0 * dfn
        m = 1.0 * dfd
        lPx = (m / 2 * np.log(m) + n / 2 * np.log(n) + sc.xlogy(n / 2 - 1, x)
               - (((n + m) / 2) * np.log(m + n * x) + sc.betaln(n / 2, m / 2)))
        return lPx

    def _cdf(self, x, dfn, dfd):
        return sc.fdtr(dfn, dfd, x)

    def _sf(self, x, dfn, dfd):
        return sc.fdtrc(dfn, dfd, x)

    def _ppf(self, q, dfn, dfd):
        return sc.fdtri(dfn, dfd, q)

    def _stats(self, dfn, dfd):
        v1, v2 = 1. * dfn, 1. * dfd
        v2_2, v2_4, v2_6, v2_8 = v2 - 2., v2 - 4., v2 - 6., v2 - 8.

        mu = lw(
            v2 > 2, (v2, v2_2),
            lambda v2, v2_2: v2 / v2_2,
            np.inf)

        mu2 = lw(
            v2 > 4, (v1, v2, v2_2, v2_4),
            lambda v1, v2, v2_2, v2_4:
            2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2**2 * v2_4),
            np.inf)

        g1 = lw(
            v2 > 6, (v1, v2_2, v2_4, v2_6),
            lambda v1, v2_2, v2_4, v2_6:
            (2 * v1 + v2_2) / v2_6 * np.sqrt(v2_4 / (v1 * (v1 + v2_2))),
            np.nan)
        g1 *= np.sqrt(8.)

        g2 = lw(
            v2 > 8, (g1, v2_6, v2_8),
            lambda g1, v2_6, v2_8: (8 + g1 * g1 * v2_6) / v2_8,
            np.nan)
        g2 *= 3. / 2.

        return mu, mu2, g1, g2

    def _entropy(self, dfn, dfd):
        half_dfn = 0.5 * dfn
        half_dfd = 0.5 * dfd
        half_sum = 0.5 * (dfn + dfd)

        return (np.log(dfd) - np.log(dfn) + sc.betaln(half_dfn, half_dfd)
                + (1 - half_dfn) * sc.psi(half_dfn) - (1 + half_dfd)
                * sc.psi(half_dfd) + half_sum * sc.psi(half_sum))
