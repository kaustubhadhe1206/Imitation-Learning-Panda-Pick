"""Combine all sweep results into the report table and comparison figures (Part 4).

Reads results/runs/<condition>_s<seed>/, results/teacher_eval/ and the demo
collection summary, and writes:
  results/summary_table.md / .csv   the table the homework asks for
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

RESULTS_DIR = utils.REPO_ROOT / "results"


def load_sweep_results(sweep):
  """For every condition, the finished runs (seed, eval summary, train summary, loss curve)."""
  results_by_condition = {}
  for condition in sweep["conditions"]:
    finished_runs = []
    for seed in sweep["seeds"]:
      run_folder = RESULTS_DIR / "runs" / f"{condition['name']}_s{seed}"
      if not (run_folder / "eval_summary.json").exists():
        continue
      loss_rows = []
      if (run_folder / "loss.csv").exists():
        with open(run_folder / "loss.csv") as f:
          loss_rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
      train_summary_path = run_folder / "train_summary.json"
      finished_runs.append({
          "seed": seed,
          "eval": utils.load_json(run_folder / "eval_summary.json"),
          "train": utils.load_json(train_summary_path) if train_summary_path.exists() else None,
          "loss": loss_rows,
      })
    results_by_condition[condition["name"]] = {"num_demos": condition["num_demos"], "runs": finished_runs}
  return results_by_condition


def build_summary_rows(results_by_condition, teacher_eval, collection_summary):
  rows = []
  for condition_name, condition in results_by_condition.items():
    if not condition["runs"]:
      continue
    success_rates = np.array([run["eval"]["success_rate"] for run in condition["runs"]])
    total_success = sum(run["eval"]["num_success"] for run in condition["runs"])
    total_episodes = sum(run["eval"]["num_episodes"] for run in condition["runs"])
    ci_low, ci_high = utils.wilson_confidence_interval(total_success, total_episodes)
    rows.append({
        "condition": condition_name,
        "num_demos": condition["num_demos"],
        "teacher_success_collection": collection_summary.get("teacher_success_rate") if collection_summary else None,
        "teacher_success_eval": teacher_eval.get("success_rate") if teacher_eval else None,
        "student_success_mean": float(success_rates.mean()),
        "student_success_std": float(success_rates.std(ddof=1)) if len(success_rates) > 1 else 0.0,
        "student_success_per_seed": success_rates.tolist(),
        "pooled_success": f"{total_success}/{total_episodes}",
        "pooled_ci95": [ci_low, ci_high],
        "num_seeds": len(success_rates),
        "eval_episodes_per_seed": condition["runs"][0]["eval"]["num_episodes"],
    })
  return rows


def write_markdown_table(rows, path):
  def as_percent(value):
    return "-" if value is None else f"{100 * value:.0f}%"
  lines = [
      "| Condition | Number of demonstrations | Teacher success rate (collection / eval seeds) "
      "| Student success rate (mean ± std over seeds) | Pooled [95% CI] | Evaluation episodes |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    lines.append(
        f"| {row['condition']} | {row['num_demos']} | {as_percent(row['teacher_success_collection'])} / "
        f"{as_percent(row['teacher_success_eval'])} | {100 * row['student_success_mean']:.1f}% ± "
        f"{100 * row['student_success_std']:.1f}% ({row['num_seeds']} seeds) | {row['pooled_success']} "
        f"[{100 * row['pooled_ci95'][0]:.0f}-{100 * row['pooled_ci95'][1]:.0f}%] | "
        f"{row['eval_episodes_per_seed']} per seed |")
  path.write_text("\n".join(lines) + "\n")


def plot_success_vs_num_demos(results_by_condition, teacher_eval, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(5.5, 3.8))
  num_demos_list, mean_rates, std_rates = [], [], []
  for condition in results_by_condition.values():
    if not condition["runs"]:
      continue
    rates = np.array([run["eval"]["success_rate"] for run in condition["runs"]])
    # small dots = individual seeds
    ax.scatter(np.full(len(rates), condition["num_demos"]), 100 * rates, s=18,
               color=plotting.COLORS[0], alpha=0.45, zorder=3)
    num_demos_list.append(condition["num_demos"])
    mean_rates.append(100 * rates.mean())
    std_rates.append(100 * rates.std(ddof=1) if len(rates) > 1 else 0.0)
  order = np.argsort(num_demos_list)
  num_demos_list = np.array(num_demos_list)[order]
  mean_rates = np.array(mean_rates)[order]
  std_rates = np.array(std_rates)[order]
  ax.errorbar(num_demos_list, mean_rates, yerr=std_rates, color=plotting.COLORS[0], marker="o",
              capsize=4, zorder=4, label="student (mean ± std over seeds)")
  for num_demos, mean_rate in zip(num_demos_list, mean_rates):
    ax.annotate(f"{mean_rate:.0f}%", (num_demos, mean_rate), textcoords="offset points", xytext=(8, -4),
                color=plotting.TEXT_COLOR, fontsize=9)
  if teacher_eval:
    teacher_rate = 100 * teacher_eval["success_rate"]
    ax.axhline(teacher_rate, color=plotting.REFERENCE_COLOR, ls="--", lw=1.2)
    ax.annotate(f"teacher {teacher_rate:.0f}%", (num_demos_list[0], teacher_rate), textcoords="offset points",
                xytext=(0, 5), color=plotting.SECONDARY_TEXT_COLOR, fontsize=9)
  ax.set_xscale("log", base=2)
  ax.set_xticks(num_demos_list, [str(n) for n in num_demos_list])
  ax.set_ylim(0, 105)
  ax.set(xlabel="number of demonstrations (log scale)", ylabel="success rate [%]",
         title="Student success vs. dataset size")
  ax.legend(loc="lower right", fontsize=8)
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_success_vs_threshold(results_by_condition, teacher_eval, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(5.5, 3.8))
  for color_index, condition in enumerate(results_by_condition.values()):
    if not condition["runs"]:
      continue
    thresholds = sorted(condition["runs"][0]["eval"]["success_at_threshold"], key=float)
    rates = [100 * np.mean([run["eval"]["success_at_threshold"][t] for run in condition["runs"]])
             for t in thresholds]
    ax.plot([100 * float(t) for t in thresholds], rates, marker="o", color=plotting.COLORS[color_index],
            label=f"{condition['num_demos']} demos")
  if teacher_eval:
    thresholds = sorted(teacher_eval["success_at_threshold"], key=float)
    ax.plot([100 * float(t) for t in thresholds], [100 * teacher_eval["success_at_threshold"][t] for t in thresholds],
            color=plotting.REFERENCE_COLOR, ls="--", marker="s", label="teacher")
  ax.axvline(5, color=plotting.GRID_COLOR, lw=3, zorder=0)   # the threshold we use
  ax.set_ylim(0, 105)
  ax.set(xlabel="success threshold on final box-target distance [cm]", ylabel="success rate [%]",
         title="Sensitivity to the success threshold")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_loss_curves(results_by_condition, path):
  plt = plotting.plt
  fig, ax = plt.subplots(figsize=(6.5, 3.8))
  for color_index, condition in enumerate(results_by_condition.values()):
    for run_index, run in enumerate(condition["runs"]):
      if run["loss"]:
        ax.plot([row["step"] for row in run["loss"]], [row["loss"] for row in run["loss"]], lw=1.2,
                color=plotting.COLORS[color_index], alpha=0.8,
                label=f"{condition['num_demos']} demos" if run_index == 0 else None)
  ax.set_yscale("log")
  ax.set(xlabel="gradient step", ylabel="noise-prediction MSE (train)", title="Training loss (all seeds)")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_failure_modes(results_by_condition, teacher_eval, path):
  plt = plotting.plt
  bar_labels, counts = [], []
  if teacher_eval:
    bar_labels.append("teacher")
    counts.append([teacher_eval["failure_modes"].get(mode, 0) for mode in FAILURE_MODES])
  for condition in results_by_condition.values():
    if condition["runs"]:
      bar_labels.append(f"{condition['num_demos']} demos")
      counts.append([sum(run["eval"]["failure_modes"].get(mode, 0) for run in condition["runs"])
                     for mode in FAILURE_MODES])
  counts = np.array(counts, dtype=float)
  percents = 100 * counts / counts.sum(axis=1, keepdims=True)
  fig, ax = plt.subplots(figsize=(8, 0.6 * len(bar_labels) + 1.8))
  bar_start = np.zeros(len(bar_labels))
  for mode_index, mode in enumerate(FAILURE_MODES):
    if percents[:, mode_index].sum() == 0:
      continue
    ax.barh(bar_labels, percents[:, mode_index], left=bar_start, color=plotting.COLORS[mode_index],
            edgecolor="white", linewidth=2, label=mode.replace("_", " "))
    for row, (start, width) in enumerate(zip(bar_start, percents[:, mode_index])):
      if width >= 8:   # only label segments big enough to fit the text
        ax.text(start + width / 2, row, f"{width:.0f}%", ha="center", va="center", fontsize=8, color="white")
    bar_start += percents[:, mode_index]
  ax.set_xlim(0, 100)
  ax.invert_yaxis()
  ax.set(xlabel="episodes [%] (pooled over seeds)", title="Episode outcomes by failure mode")
  ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_action_smoothness(results_by_condition, teacher_eval, path):
  plt = plotting.plt
  bar_labels, jerk_within_chunk, jerk_at_boundary = [], [], []
  if teacher_eval and "jerk_within" in teacher_eval:
    bar_labels.append("teacher")
    jerk_within_chunk.append(teacher_eval["jerk_within"])
    jerk_at_boundary.append(teacher_eval["jerk_boundary"])
  for condition in results_by_condition.values():
    if condition["runs"]:
      bar_labels.append(f"{condition['num_demos']} demos")
      jerk_within_chunk.append(np.mean([run["eval"]["jerk_within"] for run in condition["runs"]]))
      jerk_at_boundary.append(np.mean([run["eval"]["jerk_boundary"] for run in condition["runs"]]))
  x = np.arange(len(bar_labels))
  fig, ax = plt.subplots(figsize=(6.5, 3.6))
  ax.bar(x - 0.18, jerk_within_chunk, 0.34, color=plotting.COLORS[0], label="within a chunk")
  ax.bar(x + 0.18, jerk_at_boundary, 0.34, color=plotting.COLORS[1], label="at chunk boundary")
  ax.set_xticks(x, bar_labels)
  ax.set(ylabel="mean ||a_t - a_{t-1}||", title="Action discontinuity: within vs. between chunks")
  ax.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sweep", default="configs/sweep.yaml")
  args = parser.parse_args()
  plotting.use_report_style()

  sweep = utils.load_config(args.sweep)
  results_by_condition = load_sweep_results(sweep)
  teacher_eval_path = RESULTS_DIR / "teacher_eval" / "eval_summary.json"
  teacher_eval = utils.load_json(teacher_eval_path) if teacher_eval_path.exists() else None
  collection_path = RESULTS_DIR / "demos" / "collection_baseline.json"
  collection_summary = utils.load_json(collection_path) if collection_path.exists() else None

  rows = build_summary_rows(results_by_condition, teacher_eval, collection_summary)
  if not rows:
    print("No finished runs found in results/runs/.")
    return
  with open(RESULTS_DIR / "summary_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  write_markdown_table(rows, RESULTS_DIR / "summary_table.md")
  utils.save_json({"table": rows, "teacher_eval": teacher_eval}, RESULTS_DIR / "summary.json")

  plot_success_vs_num_demos(results_by_condition, teacher_eval, RESULTS_DIR / "fig_success_vs_demos.png")
  plot_success_vs_threshold(results_by_condition, teacher_eval, RESULTS_DIR / "fig_success_vs_threshold.png")
  plot_loss_curves(results_by_condition, RESULTS_DIR / "fig_loss_curves.png")
  plot_failure_modes(results_by_condition, teacher_eval, RESULTS_DIR / "fig_failure_modes.png")
  plot_action_smoothness(results_by_condition, teacher_eval, RESULTS_DIR / "fig_smoothness.png")

  print((RESULTS_DIR / "summary_table.md").read_text())
  print(json.dumps({row["condition"]: row["student_success_per_seed"] for row in rows}, indent=2))


if __name__ == "__main__":
  main()
