from matplotlib.axes import Axes
from matplotlib.figure import Figure

FigAx = tuple[Figure, Axes]
Fig2Ax = tuple[Figure, tuple[Axes, Axes]]


def grid(ax: Axes) -> None:
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)  # Grid behind bars
