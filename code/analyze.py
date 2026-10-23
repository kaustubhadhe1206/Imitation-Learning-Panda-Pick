"""Aggregate sweep results into the report table and comparison figures.

Reads results/runs/<cond>_s<seed>/, results/teacher_eval/ and the collection
summary; writes
  results/summary_table.md / .csv   the table requested in the homework
  results/summary.json
  results/fig_success_vs_demos.png
  results/fig_success_vs_threshold.png
  results/fig_loss_curves.png
  results/fig_failure_modes.png
  results/fig_smoothness.png

Usage:
  python code/analyze.py [--sweep configs/sweep.yaml]
"""

import argparse
import csv
import json

import numpy as np

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.rollout import FAILURE_MODES

RESULTS = utils.REPO_ROOT / "results"


def load_runs(sweep):
  runs = {}
  for cond in sweep["conditions"]:
    entries = []
    for seed in sweep["seeds"]:
      d = RESULTS / "runs" / f"{cond['name']}_s{seed}"
      if (d / "eval_summary.json").exists():
        loss = []
        if (d / "loss.csv").exists():
          with open(d / "loss.csv") as f:
            loss = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        entries.append({"seed": seed, "eval": utils.load_json(d / "eval_summary.json"),
                        "train": utils.load_json(d / "train_summary.json")
                        if (d / "train_summary.json").exists() else None, "loss": loss})
    runs[cond["name"]] = {"num_demos": cond["num_demos"], "runs": entries}
  return runs


def build_table(runs, teacher_eval, collection):
  rows = []
  for name, c in runs.items():
    if not c["runs"]:
      continue
    rates = np.array([r["eval"]["success_rate"] for r in c["runs"]])
    k = sum(r["eval"]["num_success"] for r in c["runs"])
    n = sum(r["eval"]["num_episodes"] for r in c["runs"])
    lo, hi = utils.wilson_interval(k, n)
    rows.append({
        "condition": name,
        "num_demos": c["num_demos"],
        "teacher_success_collection": collection.get("teacher_success_rate") if collection else None,
        "teacher_success_eval": teacher_eval.get("success_rate") if teacher_eval else None,
        "student_success_mean": float(rates.mean()),
        "student_success_std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
        "student_success_per_seed": rates.tolist(),
        "pooled_success": f"{k}/{n}",
        "pooled_ci95": [lo, hi],
        "num_seeds": len(rates),
        "eval_episodes_per_seed": c["runs"][0]["eval"]["num_episodes"],
    })
  return rows


def write_markdown(rows, path):
  def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"
  lines = [
      "| Condition | Number of demonstrations | Teacher success rate (collection / eval seeds) "
      "| Student success rate (mean ± std over seeds) | Pooled [95% CI] | Evaluation episodes |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for r in rows:
    lines.append(
        f"| {r['condition']} | {r['num_demos']} | {pct(r['teacher_success_collection'])} / "
        f"{pct(r['teacher_success_eval'])} | {100 * r['student_success_mean']:.1f}% ± "
        f"{100 * r['student_success_std']:.1f}% ({r['num_seeds']} seeds) | {r['pooled_success']} "
        f"[{100 * r['pooled_ci95'][0]:.0f}-{100 * r['pooled_ci95'][1]:.0f}%] | "
        f"{r['eval_episodes_per_seed']} per seed |")
  path.write_text("\n".join(lines) + "\n")


def fig_success_vs_demos(runs, teacher_eval, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(5.5, 3.8))
  xs, means, stds = [], [], []
  for c in runs.values():
    if not c["runs"]:
      continue
    rates = np.array([r["eval"]["success_rate"] for r in c["runs"]])
    ax.scatter(np.full(len(rates), c["num_demos"]), 100 * rates, s=18, color=plotting.SERIES[0],
               alpha=0.45, zorder=3)
    xs.append(c["num_demos"])
    means.append(100 * rates.mean())
    stds.append(100 * rates.std(ddof=1) if len(rates) > 1 else 0.0)
  order = np.argsort(xs)
  xs, means, stds = np.array(xs)[order], np.array(means)[order], np.array(stds)[order]
  ax.errorbar(xs, means, yerr=stds, color=plotting.SERIES[0], marker="o", capsize=4, zorder=4,
              label="student (mean ± std over seeds)")
  for x, m in zip(xs, means):
    ax.annotate(f"{m:.0f}%", (x, m), textcoords="offset points", xytext=(8, -4), color=plotting.TEXT, fontsize=9)
  if teacher_eval:
    ax.axhline(100 * teacher_eval["success_rate"], color=plotting.NEUTRAL, ls="--", lw=1.2)
    ax.annotate(f"teacher {100 * teacher_eval['success_rate']:.0f}%", (xs[0], 100 * teacher_eval["success_rate"]),
                textcoords="offset points", xytext=(0, 5), color=plotting.TEXT_2, fontsize=9)
  ax.set_xscale("log", base=2)
  ax.set_xticks(xs, [str(x) for x in xs])
  ax.set_ylim(0, 105)
  ax.set(xlabel="number of demonstrations (log scale)", ylabel="success rate [%]",
         title="Student success vs. dataset size")
  ax.legend(loc="lower right", fontsize=8)
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_success_vs_threshold(runs, teacher_eval, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(5.5, 3.8))
  for i, (name, c) in enumerate(runs.items()):
    if not c["runs"]:
      continue
    ths = sorted(c["runs"][0]["eval"]["success_at_threshold"], key=float)
    vals = [100 * np.mean([r["eval"]["success_at_threshold"][t] for r in c["runs"]]) for t in ths]
    ax.plot([100 * float(t) for t in ths], vals, marker="o", color=plotting.SERIES[i],
            label=f"{c['num_demos']} demos")
  if teacher_eval:
    ths = sorted(teacher_eval["success_at_threshold"], key=float)
    ax.plot([100 * float(t) for t in ths], [100 * teacher_eval["success_at_threshold"][t] for t in ths],
            color=plotting.NEUTRAL, ls="--", marker="s", label="teacher")
  ax.axvline(5, color=plotting.GRID, lw=3, zorder=0)
  ax.set_ylim(0, 105)
  ax.set(xlabel="success threshold on final box-target distance [cm]", ylabel="success rate [%]",
         title="Sensitivity to the success threshold")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_loss_curves(runs, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(6.5, 3.8))
  for i, (name, c) in enumerate(runs.items()):
    for j, r in enumerate(c["runs"]):
      if r["loss"]:
        ax.plot([x["step"] for x in r["loss"]], [x["loss"] for x in r["loss"]], lw=1.2,
                color=plotting.SERIES[i], alpha=0.8, label=f"{c['num_demos']} demos" if j == 0 else None)
  ax.set_yscale("log")
  ax.set(xlabel="gradient step", ylabel="noise-prediction MSE (train)", title="Training loss (all seeds)")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_failure_modes(runs, teacher_eval, path):
  plt = plotting.plt
  labels, counts = [], []
  if teacher_eval:
    labels.append("teacher")
    counts.append([teacher_eval["failure_modes"].get(m, 0) for m in FAILURE_MODES])
  for c in runs.values():
    if c["runs"]:
      labels.append(f"{c['num_demos']} demos")
      counts.append([sum(r["eval"]["failure_modes"].get(m, 0) for r in c["runs"]) for m in FAILURE_MODES])
  counts = np.array(counts, dtype=float)
  frac = 100 * counts / counts.sum(axis=1, keepdims=True)
  fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.8))
  left = np.zeros(len(labels))
  for j, m in enumerate(FAILURE_MODES):
    if frac[:, j].sum() == 0:
      continue
    ax.barh(labels, frac[:, j], left=left, color=plotting.SERIES[j], edgecolor="white", linewidth=2,
            label=m.replace("_", " "))
    for y, (l, w) in enumerate(zip(left, frac[:, j])):
      if w >= 8:
        ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8, color="white")
    left += frac[:, j]
  ax.set_xlim(0, 100)
  ax.invert_yaxis()
  ax.set(xlabel="episodes [%] (pooled over seeds)", title="Episode outcomes by failure mode")
  ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_smoothness(runs, teacher_eval, path):
  plt = plotting.plt
  labels, within, boundary = [], [], []
  if teacher_eval and "jerk_within" in teacher_eval:
    labels.append("teacher")
    within.append(teacher_eval["jerk_within"])
    boundary.append(teacher_eval["jerk_boundary"])
  for c in runs.values():
    if c["runs"]:
      labels.append(f"{c['num_demos']} demos")
      within.append(np.mean([r["eval"]["jerk_within"] for r in c["runs"]]))
      boundary.append(np.mean([r["eval"]["jerk_boundary"] for r in c["runs"]]))
  x = np.arange(len(labels))
  fig, ax = plt.subplots(figsize=(6.5, 3.6))
  ax.bar(x - 0.18, within, 0.34, color=plotting.SERIES[0], label="within a chunk")
  ax.bar(x + 0.18, boundary, 0.34, color=plotting.SERIES[1], label="at chunk boundary")
  ax.set_xticks(x, labels)
  ax.set(ylabel="mean ||a_t - a_{t-1}||", title="Action discontinuity: within vs. between chunks")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save(fig, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sweep", default="configs/sweep.yaml")
  args = parser.parse_args()
  plotting.setup()

  sweep = utils.load_config(args.sweep)
  runs = load_runs(sweep)
  teacher_eval = (utils.load_json(RESULTS / "teacher_eval" / "eval_summary.json")
                  if (RESULTS / "teacher_eval" / "eval_summary.json").exists() else None)
  coll_path = RESULTS / "demos" / "collection_baseline.json"
  collection = utils.load_json(coll_path) if coll_path.exists() else None

  rows = build_table(runs, teacher_eval, collection)
  if not rows:
    print("No finished runs found under results/runs/.")
    return
  with open(RESULTS / "summary_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  write_markdown(rows, RESULTS / "summary_table.md")
  utils.save_json({"table": rows, "teacher_eval": teacher_eval}, RESULTS / "summary.json")

  fig_success_vs_demos(runs, teacher_eval, RESULTS / "fig_success_vs_demos.png")
  fig_success_vs_threshold(runs, teacher_eval, RESULTS / "fig_success_vs_threshold.png")
  fig_loss_curves(runs, RESULTS / "fig_loss_curves.png")
  fig_failure_modes(runs, teacher_eval, RESULTS / "fig_failure_modes.png")
  fig_smoothness(runs, teacher_eval, RESULTS / "fig_smoothness.png")

  print((RESULTS / "summary_table.md").read_text())
  print(json.dumps({r["condition"]: r["student_success_per_seed"] for r in rows}, indent=2))


if __name__ == "__main__":
  main()
