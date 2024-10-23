from dataclasses import dataclass

import pandas as pd

from absorpig._plot_utils import Fig2Ax, grid


@dataclass
class UserInput:
    absorption_spectrum: pd.Series
    chl_concentration_g_per_liter: float
    pigment_spectrum: pd.DataFrame
    diameter_distribution: pd.Series


@dataclass
class Cell:
    absorption_spectrum: pd.Series
    chl_concentration_g_per_liter: float

    # lognorm params
    mu: float
    B: float
    sig: float


@dataclass
class Results:
    absorption_spectrum: pd.Series
    optimal_pigment_spectrum: pd.Series
    pigment_composition: pd.Series

    def plot(self) -> Fig2Ax:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(7, 4),
            layout="constrained",
        )

        # Original cell spectrum vs fitted spectrum
        ax1.plot(
            self.absorption_spectrum,
            label="Cell spectrum",
            c="k",
            linewidth=3,
        )
        ax1.plot(
            self.optimal_pigment_spectrum,
            label="Pigments: Optimized conc",
            ls="-",
        )
        ax1.set(
            xlabel="Wavelength / nm",
            ylabel=r"Absorption coefficient / $\frac{m^2}{mg Chl}$",
        )
        ax1.legend()
        ax1.grid(visible=True)

        self.pigment_composition.plot(
            kind="bar",
            ax=ax2,
            ylabel="Concentration / UNIT???",
        )
        grid(ax2)
        return fig, (ax1, ax2)
