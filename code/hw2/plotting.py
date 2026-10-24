"""Common matplotlib style for all report figures."""

import matplotlib

matplotlib.use("Agg")   # no screen needed (Colab / cluster)
import matplotlib.pyplot as plt  # noqa: E402

# Line/bar colors, always used in this order (this order is colorblind-friendly).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_COLOR = "#0b0b0b"
SECONDARY_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
REFERENCE_COLOR = "#8a8985"   # gray for reference lines like the teacher's success rate


def use_report_style():
  plt.rcParams.update({
      "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
      "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
      "axes.edgecolor": SECONDARY_TEXT_COLOR, "axes.labelcolor": TEXT_COLOR, "axes.titlecolor": TEXT_COLOR,
      "xtick.color": SECONDARY_TEXT_COLOR, "ytick.color": SECONDARY_TEXT_COLOR,
      "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
      "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6,
      "legend.frameon": False, "axes.prop_cycle": matplotlib.cycler(color=COLORS),
  })


def save_figure(fig, path):
  from pathlib import Path
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path)
  plt.close(fig)
  return path
