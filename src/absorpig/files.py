from pathlib import Path

DATADIR = Path(__file__).parent / "data"

default_pigment_spectrum = DATADIR / "pigment_spectra_raw.csv"
default_pigment_shifts = DATADIR / "pigment_shifts.csv"
default_pigment_guess = DATADIR / "pigment_concentrations.csv"
cell_diameter_distribution = DATADIR / "cell_diameter_distribution.csv"
qexp_integralfraction = DATADIR / "Qexp_itegralfraction_subs.pickle"
