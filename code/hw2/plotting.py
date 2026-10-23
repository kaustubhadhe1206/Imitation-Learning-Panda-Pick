"""Shared matplotlib style for report figures (light theme, print-friendly)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (colorblind-validated order; never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8985"   # reference lines (e.g. teacher), not a data series


def setup():
  plt.rcParams.update({
      "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
      "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
      "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
      "xtick.color": TEXT_2, "ytick.color": TEXT_2,
      "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
      "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6,
      "legend.frameon": False, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
  })


def save(fig, path):
  from pathlib import Path
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path)
  plt.close(fig)
  return path
