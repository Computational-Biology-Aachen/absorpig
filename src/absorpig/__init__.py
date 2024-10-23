from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import quad_vec
from scipy.optimize import OptimizeResult, fsolve, minimize
from scipy.stats import lognorm
from sympy import Symbol, exp, integrate, lambdify, oo, pi, symbols
from sympy.stats import LogNormal, density

from absorpig import files
from absorpig.data import Cell, Results, UserInput

if TYPE_CHECKING:
    from collections.abc import Callable

    from sympy.core.mul import Mul

AbsorptionSpectrum: TypeAlias = pd.Series
Array = npt.NDArray[np.float64]


def pigment_shift(
    pigment_spectra: pd.DataFrame,
    shift_values: pd.Series,
) -> pd.DataFrame:
    Pigm_shift = {}
    for pigm, wl in pigment_spectra.T.iterrows():
        wl.index = wl.index + shift_values[pigm]  # type: ignore
        Pigm_shift[pigm] = wl

    return pd.concat(Pigm_shift, axis=1)


def _get_spectrum_residuals(
    cell_spectrum: AbsorptionSpectrum,
    pigment_spectrum: Array,
    scale: float = 1e3,
) -> Array:
    return np.sqrt(np.mean((cell_spectrum - pigment_spectrum) ** 2)) * scale


def _get_pigment_spectrum(
    pigment_concentrations: Array,
    pigment_spectra: pd.DataFrame,
) -> Array:
    return np.dot(pigment_spectra, pigment_concentrations)


def _get_concentration_residuals(
    pigment_concentrations: Array,
    pigment_spectra: pd.DataFrame,
    cell_spectrum: pd.Series,
) -> Array:
    # Get the spectrum with the given concentrations
    pigment_spectrum = _get_pigment_spectrum(pigment_concentrations, pigment_spectra)

    # Get the residuals
    return _get_spectrum_residuals(cell_spectrum, pigment_spectrum)


def fit_pigments_to_spectrum(
    cell_spectrum: pd.Series,
    pigment_spectra: pd.DataFrame,
    concentration_guess: pd.Series,
) -> OptimizeResult:
    bounds = [(0, None) for i in concentration_guess]

    return minimize(
        fun=_get_concentration_residuals,
        x0=concentration_guess,
        args=(pigment_spectra, cell_spectrum),
        bounds=bounds,
    )


def _calc_rho_n(d: float, n: Array, lam: Array) -> Array:
    return 4 * d * n * np.pi / (lam / 1000)


def _cell_size_residuals(fun_param: list, data: pd.Series, fun: Callable) -> float:
    """Calculate the residuals for a

    Args:
        fun_param (dict): _description_
        data (pd.Series): _description_
        fun (Callable): _description_

    Returns:
        float: _description_
    """
    sim = fun(data.index.to_numpy(), *fun_param)
    res = ((sim - data) ** 2).sum(skipna=False)

    return np.inf if np.isnan(res) else res


def _fit_cell_size(data: pd.Series, fun: Callable, fun_args0: list) -> OptimizeResult:
    """Fit a cell size distribution using a provided function.

    Args:
        data (pd.Series): Series of cell diameters and their abundance.
        fun (Callable): Function to fit to the cell size distribution.
        fun_args0 (list): Initial guess of the fitted function parameters.

    Returns:
        OptimizeResult: Result of the optimisation.
    """
    return minimize(_cell_size_residuals, x0=fun_args0, args=(data, fun))


def _lognormal_dist(
    x: Array,
    mean: float = 0.0,
    std: float = 1.0,
    scaling: float = 1.0,
) -> Array:
    return lognorm.pdf(x, s=std, scale=np.exp(mean)) * scaling


def _get_initial_lognormal_parameters(cell_size: pd.Series) -> dict[str, float]:
    sizes = cell_size.index.to_numpy()
    sizes_log = np.log(sizes)
    dist = cell_size.to_numpy()

    # Get mu and sd from the log-transformed data
    mean = float(np.average(sizes_log, weights=dist))
    std = np.sqrt(np.average((sizes_log - mean) ** 2, weights=dist))
    scaling = dist.max() / _lognormal_dist(np.exp(mean), mean=mean, std=std)
    return {"mu": mean, "sig": std, "B": scaling}


def _get_cell_diameter_distribution(
    mean_diam: float | None = None,
) -> pd.Series:
    # Read the common cell diameter distribution
    cell_diam = pd.read_csv(files.cell_diameter_distribution)
    cell_diam = cell_diam.set_index("cell_diam_um").iloc[:, 0]

    # Shift the data to correspond to the assumed mean cell size
    # Get the mean cell size assuming a log norm function
    if mean_diam is not None:
        sizes = cell_diam.index.to_numpy()
        sizes_log = np.log(sizes)
        dist = cell_diam.to_numpy()

        # Get average cell diameter
        mu = np.exp(np.average(sizes_log, weights=dist))
        cell_diam.index = cell_diam.index * (mean_diam / mu)
    return cell_diam


def make_cell(user_input: UserInput) -> Cell:
    cell_size_fit = _fit_cell_size(
        data=user_input.diameter_distribution,
        fun=_lognormal_dist,
        fun_args0=list(
            _get_initial_lognormal_parameters(user_input.diameter_distribution).values()
        ),
    )
    mu, B, sig = cell_size_fit.x

    return Cell(
        absorption_spectrum=user_input.absorption_spectrum,
        chl_concentration_g_per_liter=user_input.chl_concentration_g_per_liter,
        mu=mu,
        B=B,
        sig=sig,
    )


@dataclass
class InVivoSpectrum: ...


def _get_lognorm_symb() -> Mul:
    """Get a symbolic expression of the log-normal density function

    Returns:
        Mul: Symbolic expression of the log-normal density
    """
    b, d, mu, sig = symbols("B,d,mu,sig")
    return density(LogNormal("X", mu, sig))(d) * b  # type: ignore


def _get_rho_symbn() -> float:
    d, n, lam = symbols("d,n,lam")
    return 4 * d * n * pi / (lam / 1000)


def _get_qth_symb() -> Symbol:
    rho = Symbol("rho")
    return 1 + 2 * (exp(-rho) / rho) + 2 * ((exp(-rho) - 1) / rho**2)  # type: ignore


def _calc_qth_symbn(
    n: float,
    lam: float,
    lognorm_param: dict[str, float],
) -> float:
    b, d, mu, sig, rho, _n, _lam = symbols("B,d,mu,sig,rho,n,lam")
    lognorm_symb = _get_lognorm_symb()

    rho_symbn = _get_rho_symbn()
    Qth_n = (_get_qth_symb() - 1).subs(rho, rho_symbn)  # type: ignore

    topfun = d**2 * lognorm_symb * (Qth_n)
    botfun = d**2 * lognorm_symb

    topfun = lambdify([d, _n, _lam, mu, sig, b], topfun, "numpy")
    botfun = lambdify([d, _n, _lam, mu, sig, b], botfun, "numpy")

    topinteg = quad_vec(
        topfun, a=0, b=np.inf, args=(n, lam, *list(lognorm_param.values()))
    )
    botinteg = quad_vec(
        botfun, a=0, b=np.inf, args=(n, lam, *list(lognorm_param.values()))
    )
    frac = topinteg[0] / botinteg[0]

    return 1 + frac


def _calc_q_diff_symbn(
    n: float,
    lam: float,
    lognorm_param: dict[str, float],
    qexp: float,
) -> float:
    Qth = _calc_qth_symbn(n, lam, lognorm_param)
    return Qth - qexp


def _fit_rhos_symbn_vectorised(
    qexp: pd.Series,
    lognorm_param: dict[str, float],
    n0: float = 5e-4,
) -> pd.Series:
    lams = qexp.index.to_numpy()
    ns = cast(
        Array,
        fsolve(
            _calc_q_diff_symbn,
            x0=np.full(len(lams), n0),
            args=(np.array(lams, dtype=float), lognorm_param, qexp.to_numpy()),
        ),
    )
    rhos = _calc_rho_n(d=np.exp(lognorm_param["mu"]), n=ns, lam=lams)
    return pd.Series(rhos, index=qexp.index)


def _calc_qpexp(qexp: pd.Series, rho: pd.Series) -> pd.Series:
    return (3 / 2) * qexp / rho


def _get_qexp_integralfraction_symb() -> Mul:
    """Get a symbolic representation of the fraction of two integrated log-normal
       functions used in the package effect calculation.
       The integrations are pre-computed and saved upon the first execution of this function.

    Returns:
        Mul: Symbolic representation of a fraction needed for the package effect calculation
    """
    if (fp := files.qexp_integralfraction).exists():
        with fp.open("rb") as f:
            return pickle.load(f)

    b, d, mu, sig = symbols("B,d,mu,sig")

    # Define a symbolical Log-Normal distribution
    # (mean mu, shape sigma, scaling B)
    lognorm_symb = _get_lognorm_symb()

    # Definte the top and botton functions
    topfun = d**3 * lognorm_symb
    botfun = d**2 * lognorm_symb

    # Use substitution rule int(f) = int(g) * diff(g)=> x = exp(x)
    topfun = topfun.subs(d, exp(d)) * exp(d)
    botfun = botfun.subs(d, exp(d)) * exp(d)

    # Integration with adapted borders
    topinteg = integrate(topfun, (d, -oo, oo))
    botinteg = integrate(botfun, (d, -oo, oo))
    frac = topinteg / botinteg  # type: ignore

    with fp.open("wb") as f:
        pickle.dump(frac, f)
    return frac


def _calc_qexp(
    cell_spectrum: pd.Series, c: float, lognorm_param: dict[str, float]
) -> pd.Series:
    mu, sig, B = symbols("mu,sig,B")

    # Evaluate the Integral of the size distribution function
    frac = _get_qexp_integralfraction_symb()
    frac = frac.subs(lognorm_param).evalf()  # type: ignore
    return cell_spectrum * c * (2 / 3) * float(frac)  # type: ignore


def get_package_effect(
    cell_spectrum: pd.Series,
    chl_conc: float,
    lognorm_param: dict[str, float],
) -> pd.Series:
    qexp = _calc_qexp(
        cell_spectrum=cell_spectrum,
        c=chl_conc,
        lognorm_param=lognorm_param,
    )

    rhos = _fit_rhos_symbn_vectorised(
        qexp=qexp,
        lognorm_param=lognorm_param,
    )

    qpexp = _calc_qpexp(qexp=qexp, rho=rhos)
    qpexp[qpexp > 1] = 1

    return qpexp


def routine(
    absorption_spectrum: pd.Series,
    chl_concentration: float,
    mean_diameter: float,
    pigment_spectrum: pd.DataFrame | None = None,
    *,
    shift_spectra: bool = True,
) -> Results:
    if pigment_spectrum is None:
        pigment_spectrum = pd.read_csv(
            files.default_pigment_spectrum,
            index_col=0,
        )

    if shift_spectra:
        pigment_spectrum = pigment_shift(
            pigment_spectra=pigment_spectrum,
            shift_values=pd.Series(
                {
                    "Chla": 0,
                    "PC": -3,
                    "APC": -3,
                    "B-car": -4,
                    "Zea": 3,
                    "Ech": 3,
                    "Myx": 0,
                    "HEch": 1,
                    "P698": 2,
                    "P702": 0,
                    "P680": 0,
                    "P686": 0,
                    "Pheo": 0,
                },
            ),
        )

    cell = make_cell(
        UserInput(
            absorption_spectrum=absorption_spectrum,
            chl_concentration_g_per_liter=chl_concentration,
            pigment_spectrum=pigment_spectrum,
            diameter_distribution=_get_cell_diameter_distribution(mean_diameter),
        )
    )

    package_effect = get_package_effect(
        cell_spectrum=cell.absorption_spectrum,
        chl_conc=cell.chl_concentration_g_per_liter,
        lognorm_param={"mu": cell.mu, "B": cell.B, "sig": cell.sig},
    )

    # Spectra corrected for package effect and pigment shift
    # Only select certain package effect spectra?
    spectra = (pigment_spectrum.T * package_effect.loc[400:700]).T

    to_fit = (spectra / cell.chl_concentration_g_per_liter).loc[
        cell.absorption_spectrum.index
    ]
    res = fit_pigments_to_spectrum(
        cell_spectrum=cell.absorption_spectrum,
        pigment_spectra=to_fit,
        concentration_guess=pd.Series(
            {
                "Chla": 1800.000000,
                "PC": 350.000000,
                "APC": 112.500000,
                "B-car": 435.000000,
                "Zea": 360.000000,
                "Ech": 150.000000,
                "Myx": 195.000000,
                "HEch": 60.000000,
                "P698": 27.480916,
                "P702": 54.961832,
                "P680": 27.480916,
                "P686": 27.480916,
                "Pheo": 27.480916,
            }
        ),
    )

    return Results(
        absorption_spectrum=absorption_spectrum,
        optimal_pigment_spectrum=pd.Series(
            data=_get_pigment_spectrum(res.x, to_fit),
            index=to_fit.index,
        ),
        pigment_composition=pd.Series(
            data=res.x,
            index=pigment_spectrum.columns,
        ),
    )
