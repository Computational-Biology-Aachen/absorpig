from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Annotated, cast

import pandas as pd
import typer

from absorpig import (
    routine,
)

app = typer.Typer()

_spectrum_help = """File path to csv containing the absorption spectrum
with wavelength: absorption pairs.\n
Format is expected to be as follows:\n
\n
|   wavelength_nm |   absorption |\n
|----------------:|-------------:|\n
|             396 |       0.0211 |\n
|             397 |       0.0216 |\n
|             398 |       0.022  |\n
|             399 |       0.0224 |\n
|             400 |       0.023  |\n
"""
_chl_concentration_help = "Chlorophyll concentration in units of FIXME"
_mean_diameter_help = "Mean cell diameter in µm"
_pigment_spectra_help = """csv file with custom pigment spectrum.\n
Format is expected to be as follows:\n
|  wavelength_nm |      Chla |          PC |\n
|---------------:|----------:|------------:|\n
|            396 | 0.0168795 | 0.000181564 |\n
|            397 | 0.017119  | 0.000181041 |\n
|            398 | 0.0173586 | 0.000180518 |\n
|            399 | 0.0175981 | 0.000179995 |\n
|            400 | 0.0178377 | 0.000179472 |\n
"""
_shift_spectra_help = (
    "Whether or not standard shifts are applied to the pigment spectra"
)


@app.command()
def main(
    absorption_spectrum_csv: Annotated[
        Path,
        typer.Argument(help=_spectrum_help),
    ],
    chl_concentration: Annotated[
        float,
        typer.Argument(help=_chl_concentration_help),
    ],
    mean_diameter: Annotated[
        float,
        typer.Argument(help=_mean_diameter_help),
    ],
    pigment_spectra_csv: Annotated[
        Path | None,
        typer.Option(help=_pigment_spectra_help),
    ] = None,
    *,
    shift_spectra: Annotated[
        bool,
        typer.Option(help=_shift_spectra_help),
    ] = True,
) -> None:
    absorption_spectrum = cast(
        pd.Series,
        pd.read_csv(
            absorption_spectrum_csv,
            index_col=0,
        ).iloc[:, 0],
    )

    if pigment_spectra_csv is None:
        pigment_spectrum = None
    else:
        pigment_spectrum = pd.read_csv(
            pigment_spectra_csv,
            index_col=0,
        )

    routine(
        absorption_spectrum=absorption_spectrum,
        chl_concentration=chl_concentration,
        mean_diameter=mean_diameter,
        pigment_spectrum=pigment_spectrum,
        shift_spectra=shift_spectra,
    )


if __name__ == "__main__":
    typer.run(main)
