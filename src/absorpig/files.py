from pathlib import Path

DATADIR = Path(__file__).parent / "data"

default_pigment_spectrum = DATADIR / "pigment_spectra.csv"
cell_diameter_distribution = DATADIR / "cell_diameter_distribution.csv"
qexp_integralfraction = DATADIR / "Qexp_itegralfraction_subs.pickle"
