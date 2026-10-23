"""Visualize and summarize a demonstration dataset (Part 1 of the homework).

Outputs (default for the baseline dataset):
  results/demos/demo_stats.json        counts, length / time-to-success stats
  results/demos/fig_coverage.png       initial box / target positions
  results/demos/fig_trajectories.png   gripper paths and box-target distance
  results/demos/fig_actions.png        per-dimension action traces
  results/demos/fig_phases.png         expert phase durations
  results/demos/fig_frames.png         film strip of several demos (rendered)
  videos/expert/*.mp4                  individual demos + a 2x2 grid

Usage:
  python code/inspect_demos.py [--data data/demos_baseline.npz] [--no-render]
"""

import argparse
import json

import numpy as np

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.rollout import FAILURE_MODES, classify_failures

PHASES = ["approach", "descend", "grasp", "lift", "transport"]


def settle_step(success_step):
  """First step after which the box stays within threshold until the end (-1 if never)."""
  T1 = success_step.shape[1]
  out = np.full(len(success_step), -1)
  for i, s in enumerate(success_step):
    if s[-1]:
      bad = np.flatnonzero(~s)
      out[i] = bad[-1] + 1 if len(bad) else 0
  return np.minimum(out, T1 - 1)


def compute_stats(d):
  succ = d["success"]
  N, T = d["actions"].shape[:2]
  path_len = np.linalg.norm(np.diff(d["grip_pos"], axis=1), axis=-1).sum(axis=1)
  settle = settle_step(d["success_step"])
  first = d["first_success_step"]
  modes = classify_failures(d)
  meta = json.loads(str(d["meta"]))

  def ms(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)),
            "max": float(np.max(x))} if len(x) else None

  return {
      "num_trajectories": int(N),
      "num_successful": int(succ.sum()),
      "success_fraction": float(succ.mean()),
      "episode_length_steps": ms(np.full(N, T)),
      "first_success_step_successful": ms(first[succ]),
      "settle_step_successful": ms(settle[succ]),
      "gripper_path_length_m_successful": ms(path_len[succ]),
      "final_dist_m_successful": ms(d["dist"][succ, -1]),
      "failure_modes": {m: int((modes == m).sum()) for m in FAILURE_MODES if (modes == m).any()},
      "box_init_range": {"min": d["box_init"].min(0).tolist(), "max": d["box_init"].max(0).tolist()},
      "target_range": {"min": d["target_pos"].min(0).tolist(), "max": d["target_pos"].max(0).tolist()},
      "action_abs_mean_per_dim": np.abs(d["actions"][succ]).mean(axis=(0, 1)).tolist(),
      "action_saturation_fraction": float((np.abs(d["actions"][succ]) > 0.999).mean()),
      "ctrl_dt": 0.02,
      "teacher": {k: meta.get(k) for k in ["name", "noise_std", "teacher_success_rate"]},
  }


def fig_coverage(d, path):
  plt = plotting.plt
  succ = d["success"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
  ax = axes[0]
  for mask, label, marker, color in [(succ, "success", "o", plotting.SERIES[0]),
                                     (~succ, "failure", "x", plotting.SERIES[1])]:
    if mask.any():
      ax.scatter(d["box_init"][mask, 0], d["box_init"][mask, 1], s=22, marker=marker,
                 color=color, label=f"box start ({label})")
  ax.scatter(d["target_pos"][:, 0], d["target_pos"][:, 1], s=22, marker="^",
             facecolors="none", edgecolors=plotting.SERIES[2], label="target")
  ax.set(xlabel="x [m]", ylabel="y [m]", title="Initial conditions (top-down)", aspect="equal")
  ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
  ax = axes[1]
  ax.hist(d["target_pos"][:, 2], bins=15, color=plotting.SERIES[2], edgecolor="white", linewidth=1)
  ax.set(xlabel="target height z [m]", ylabel="episodes", title="Target height distribution")
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_trajectories(d, path, n_show=20):
  plt = plotting.plt
  succ_idx = np.flatnonzero(d["success"])[:n_show]
  fig, axes = plt.subplots(1, 3, figsize=(14, 4))
  for i in succ_idx:
    g = d["grip_pos"][i]
    axes[0].plot(g[:, 0], g[:, 1], lw=1, color=plotting.SERIES[0], alpha=0.5)
    axes[1].plot(g[:, 0], g[:, 2], lw=1, color=plotting.SERIES[0], alpha=0.5)
  axes[0].scatter(d["box_init"][succ_idx, 0], d["box_init"][succ_idx, 1], s=12, color=plotting.TEXT_2, label="box start")
  axes[0].scatter(d["target_pos"][succ_idx, 0], d["target_pos"][succ_idx, 1], s=14, marker="^",
                  color=plotting.SERIES[2], label="target")
  axes[0].set(xlabel="x [m]", ylabel="y [m]", title=f"Gripper paths, top-down ({len(succ_idx)} demos)")
  axes[0].legend(fontsize=8)
  axes[1].set(xlabel="x [m]", ylabel="z [m]", title="Gripper paths, side view")
  t = np.arange(d["dist"].shape[1]) * 0.02
  dist = d["dist"][d["success"]]
  med, lo, hi = np.median(dist, 0), np.percentile(dist, 10, 0), np.percentile(dist, 90, 0)
  axes[2].fill_between(t, lo, hi, color=plotting.SERIES[0], alpha=0.2, linewidth=0, label="10-90th pct")
  axes[2].plot(t, med, color=plotting.SERIES[0], label="median")
  axes[2].axhline(0.05, color=plotting.NEUTRAL, ls="--", lw=1.2, label="success threshold")
  axes[2].set(xlabel="time [s]", ylabel="box-target distance [m]",
              title=f"Box-target distance (successful demos, n={len(dist)})")
  axes[2].legend(fontsize=8)
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_actions(d, path, n_show=10):
  plt = plotting.plt
  idx = np.flatnonzero(d["success"])[:n_show]
  t = np.arange(d["actions"].shape[1]) * 0.02
  names = [f"joint {j + 1}" for j in range(7)] + ["gripper"]
  fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharex=True, sharey=True)
  for j, ax in enumerate(axes.flat):
    for i in idx:
      ax.plot(t, d["actions"][i, :, j], lw=1, color=plotting.SERIES[0], alpha=0.45)
    ax.set_title(names[j])
    ax.set_ylim(-1.05, 1.05)
  for ax in axes[1]:
    ax.set_xlabel("time [s]")
  for ax in axes[:, 0]:
    ax.set_ylabel("action")
  fig.suptitle(f"Expert actions for {len(idx)} successful demos (normalized delta joint targets)")
  fig.tight_layout()
  return plotting.save(fig, path)


def fig_phases(d, path):
  plt = plotting.plt
  succ = d["success"]
  dur = [(d["phase"][succ] == p).sum(axis=1) for p in range(len(PHASES))]
  fig, ax = plt.subplots(figsize=(7, 3.6))
  bp = ax.boxplot(dur, tick_labels=PHASES, patch_artist=True, widths=0.5)
  for box in bp["boxes"]:
    box.set(facecolor=plotting.SERIES[0], alpha=0.35, edgecolor=plotting.SERIES[0])
  for med in bp["medians"]:
    med.set(color=plotting.TEXT, linewidth=1.5)
  ax.set(ylabel="steps (x 0.02 s)", title="Time spent in each expert phase (successful demos)")
  fig.tight_layout()
  return plotting.save(fig, path)


def render_outputs(d, out_dir, video_dir, num_videos):
  from hw2.envs import load_env
  from hw2.render import TrajectoryRenderer, annotate, tile, write_video
  plt = plotting.plt

  env = load_env()
  fps = 25  # every 2nd state of a 50 Hz rollout -> real time
  renderer = TrajectoryRenderer(env.mj_model, height=480, width=640)
  succ_idx = np.flatnonzero(d["success"])
  fail_idx = np.flatnonzero(~d["success"])
  written = []
  for i in succ_idx[:num_videos]:
    frames = renderer.render(d["qpos"][i], d["target_pos"][i], every=2)
    frames = annotate(frames, f"expert demo seed {d['seeds'][i]} - success")
    written.append(str(write_video(video_dir / f"expert_demo_seed{d['seeds'][i]}.mp4", frames, fps)))
  for i in fail_idx[:1]:
    frames = renderer.render(d["qpos"][i], d["target_pos"][i], every=2)
    frames = annotate(frames, f"expert seed {d['seeds'][i]} - FAILURE")
    written.append(str(write_video(video_dir / f"expert_failure_seed{d['seeds'][i]}.mp4", frames, fps)))
  renderer.close()

  small = TrajectoryRenderer(env.mj_model, height=240, width=320)
  grid = [annotate(small.render(d["qpos"][i], d["target_pos"][i], every=2), f"seed {d['seeds'][i]}")
          for i in succ_idx[:4]]
  written.append(str(write_video(video_dir / "expert_demos_grid.mp4", tile(grid, 2), fps)))

  # Film strip: rows = demos, columns = time.
  times = [0, 25, 50, 75, 100, 150]
  rows = succ_idx[:4]
  fig, axes = plt.subplots(len(rows), len(times), figsize=(2.2 * len(times), 1.75 * len(rows)))
  for r, i in enumerate(rows):
    frames = small.render(d["qpos"][i][times], d["target_pos"][i])
    for c, (f, t) in enumerate(zip(frames, times)):
      ax = axes[r, c]
      ax.imshow(f)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.grid(False)
      for s in ax.spines.values():
        s.set_visible(False)
      if r == 0:
        ax.set_title(f"t = {t * 0.02:.1f} s", fontsize=9)
      if c == 0:
        ax.set_ylabel(f"seed {d['seeds'][i]}", fontsize=9)
  fig.suptitle("Expert demonstrations (red translucent box = target)", fontsize=11)
  fig.tight_layout()
  plotting.save(fig, out_dir / "fig_frames.png")
  small.close()
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data", default="data/demos_baseline.npz")
  parser.add_argument("--out", default="results/demos")
  parser.add_argument("--video-dir", default="videos/expert")
  parser.add_argument("--num-videos", type=int, default=3)
  parser.add_argument("--no-render", action="store_true")
  args = parser.parse_args()

  plotting.setup()
  out_dir = utils.resolve_path(args.out)
  with np.load(utils.resolve_path(args.data)) as z:
    d = {k: z[k] for k in z.files}

  stats = compute_stats(d)
  utils.save_json(stats, out_dir / "demo_stats.json")
  fig_coverage(d, out_dir / "fig_coverage.png")
  fig_trajectories(d, out_dir / "fig_trajectories.png")
  fig_actions(d, out_dir / "fig_actions.png")
  fig_phases(d, out_dir / "fig_phases.png")
  videos = [] if args.no_render else render_outputs(d, out_dir, utils.resolve_path(args.video_dir), args.num_videos)

  print("\n==== DEMO STATS (paste this back) ====")
  print(json.dumps({k: stats[k] for k in [
      "num_trajectories", "num_successful", "success_fraction", "episode_length_steps",
      "first_success_step_successful", "settle_step_successful", "gripper_path_length_m_successful",
      "final_dist_m_successful", "failure_modes", "action_saturation_fraction"]}, indent=2))
  print("figures ->", out_dir)
  for v in videos:
    print("video ->", v)


if __name__ == "__main__":
  main()
