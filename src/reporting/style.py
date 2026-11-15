"""Shared labels and chart style for the results notebooks (03, 04) and the presentation figures."""
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES = PROJECT_ROOT / "results" / "figures"

# Source product name (as published) -> English label used in charts and tables.
LABELS = {
    "Картопля": "Potato",
    "Морква": "Carrot",
    "Яйця": "Eggs",
    "Молоко пастеризоване жирністю до 2,6% включно": "Pasteurized milk",
    "Бензин А-95": "A-95 gasoline",
    "Олія соняшникова": "Sunflower oil",
}

# One colour per model in every chart. Benchmarks are greys so the fitted models stand out.
MODEL_COLORS = {
    "naive": "#52514e",
    "drift": "#a9a8a2",
    "seasonal_naive": "#8a8983",
    "snaive+yoy": "#c3c2b7",
    "STL+drift": "#1baf7a",
    "ETS": "#6c3ce9",
    "SARIMA": "#f07f2d",
}
BENCHMARKS = ["naive", "drift", "seasonal_naive", "snaive+yoy"]
ACTUAL_COLOR = "#0b0b0b"
MUTED = "#52514e"


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": MUTED,
        "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def save(fig, name):
    """Write the figure to results/figures/<name>.png, where the presentation picks it up."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
