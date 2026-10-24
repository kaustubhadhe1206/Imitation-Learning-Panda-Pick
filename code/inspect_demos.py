"""Look at the demonstration dataset: statistics, figures and videos (Part 1).

Outputs (for the baseline dataset):
  results/demos/demo_stats.json        counts, episode length, time-to-success
  results/demos/fig_coverage.png       where the box and target start
  results/demos/fig_trajectories.png   gripper paths and box-to-target distance
  results/demos/fig_actions.png        expert actions for each joint
  results/demos/fig_phases.png         time spent in each expert phase
  results/demos/fig_frames.png         frames from several demos
  videos/expert/*.mp4                  single demos and a 2x2 grid

Usage:
  python code/inspect_demos.py [--data data/demos_baseline.npz] [--no-render]
"""

import argparse
import json

import numpy as np

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.rollout import FAILURE_MODES, label_failure_modes

PHASE_NAMES = ["approach", "descend", "grasp", "lift", "transport"]
VIDEO_FPS = 25          # we keep every 2nd state of a 50 Hz episode, so 25 fps is real time
CONTROL_DT = 0.02       # seconds per env step


def find_settle_steps(success_at_step):
  """First step after which the box stays at the target until the end (-1 if it never settles)."""
  num_positions = success_at_step.shape[1]
  settle_steps = np.full(len(success_at_step), -1)
  for i, success_flags in enumerate(success_at_step):
    if success_flags[-1]:
      not_at_target = np.flatnonzero(~success_flags)
      settle_steps[i] = not_at_target[-1] + 1 if len(not_at_target) else 0
  return np.minimum(settle_steps, num_positions - 1)


def compute_dataset_stats(demos):
  successful = demos["success"]
  num_episodes, num_steps = demos["actions"].shape[:2]
  gripper_path_length = np.linalg.norm(np.diff(demos["grip_pos"], axis=1), axis=-1).sum(axis=1)
  settle_steps = find_settle_steps(demos["success_step"])
  failure_modes = label_failure_modes(demos)
  collection_info = json.loads(str(demos["meta"]))

  def mean_std_min_max(values):
    if not len(values):
      return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}

  successful_actions = demos["actions"][successful]
  return {
      "num_trajectories": int(num_episodes),
      "num_successful": int(successful.sum()),
      "success_fraction": float(successful.mean()),
      "episode_length_steps": mean_std_min_max(np.full(num_episodes, num_steps)),
      "first_success_step_successful": mean_std_min_max(demos["first_success_step"][successful]),
      "settle_step_successful": mean_std_min_max(settle_steps[successful]),
      "gripper_path_length_m_successful": mean_std_min_max(gripper_path_length[successful]),
      "final_dist_m_successful": mean_std_min_max(demos["dist"][successful, -1]),
      "failure_modes": {mode: int((failure_modes == mode).sum())
                        for mode in FAILURE_MODES if (failure_modes == mode).any()},
      "box_init_range": {"min": demos["box_init"].min(0).tolist(), "max": demos["box_init"].max(0).tolist()},
      "target_range": {"min": demos["target_pos"].min(0).tolist(), "max": demos["target_pos"].max(0).tolist()},
      "action_abs_mean_per_dim": np.abs(successful_actions).mean(axis=(0, 1)).tolist(),
      "action_saturation_fraction": float((np.abs(successful_actions) > 0.999).mean()),
      "ctrl_dt": CONTROL_DT,
      "teacher": {key: collection_info.get(key) for key in ["name", "noise_std", "teacher_success_rate"]},
  }


def plot_initial_conditions(demos, path):
  plt = plotting.plt
  successful = demos["success"]
  fig, (ax_top, ax_height) = plt.subplots(1, 2, figsize=(10, 4.2))
  for mask, label, marker, color in [(successful, "success", "o", plotting.COLORS[0]),
                                     (~successful, "failure", "x", plotting.COLORS[1])]:
    if mask.any():
      ax_top.scatter(demos["box_init"][mask, 0], demos["box_init"][mask, 1], s=22, marker=marker,
                     color=color, label=f"box start ({label})")
  ax_top.scatter(demos["target_pos"][:, 0], demos["target_pos"][:, 1], s=22, marker="^",
                 facecolors="none", edgecolors=plotting.COLORS[2], label="target")
  ax_top.set(xlabel="x [m]", ylabel="y [m]", title="Initial conditions (top-down)", aspect="equal")
  ax_top.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
  ax_height.hist(demos["target_pos"][:, 2], bins=15, color=plotting.COLORS[2], edgecolor="white", linewidth=1)
  ax_height.set(xlabel="target height z [m]", ylabel="episodes", title="Target height distribution")
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_trajectories(demos, path, num_to_show=20):
  plt = plotting.plt
  shown = np.flatnonzero(demos["success"])[:num_to_show]
  fig, (ax_top, ax_side, ax_dist) = plt.subplots(1, 3, figsize=(14, 4))
  for i in shown:
    gripper_path = demos["grip_pos"][i]
    ax_top.plot(gripper_path[:, 0], gripper_path[:, 1], lw=1, color=plotting.COLORS[0], alpha=0.5)
    ax_side.plot(gripper_path[:, 0], gripper_path[:, 2], lw=1, color=plotting.COLORS[0], alpha=0.5)
  ax_top.scatter(demos["box_init"][shown, 0], demos["box_init"][shown, 1], s=12,
                 color=plotting.SECONDARY_TEXT_COLOR, label="box start")
  ax_top.scatter(demos["target_pos"][shown, 0], demos["target_pos"][shown, 1], s=14, marker="^",
                 color=plotting.COLORS[2], label="target")
  ax_top.set(xlabel="x [m]", ylabel="y [m]", title=f"Gripper paths, top-down ({len(shown)} demos)")
  ax_top.legend(fontsize=8)
  ax_side.set(xlabel="x [m]", ylabel="z [m]", title="Gripper paths, side view")

  time_s = np.arange(demos["dist"].shape[1]) * CONTROL_DT
  distances = demos["dist"][demos["success"]]
  median = np.median(distances, 0)
  low, high = np.percentile(distances, 10, 0), np.percentile(distances, 90, 0)
  ax_dist.fill_between(time_s, low, high, color=plotting.COLORS[0], alpha=0.2, linewidth=0, label="10-90th pct")
  ax_dist.plot(time_s, median, color=plotting.COLORS[0], label="median")
  ax_dist.axhline(0.05, color=plotting.REFERENCE_COLOR, ls="--", lw=1.2, label="success threshold")
  ax_dist.set(xlabel="time [s]", ylabel="box-target distance [m]",
              title=f"Box-target distance (successful demos, n={len(distances)})")
  ax_dist.legend(fontsize=8)
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_actions(demos, path, num_to_show=10):
  plt = plotting.plt
  shown = np.flatnonzero(demos["success"])[:num_to_show]
  time_s = np.arange(demos["actions"].shape[1]) * CONTROL_DT
  dimension_names = [f"joint {j + 1}" for j in range(7)] + ["gripper"]
  fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharex=True, sharey=True)
  for dim, ax in enumerate(axes.flat):
    for i in shown:
      ax.plot(time_s, demos["actions"][i, :, dim], lw=1, color=plotting.COLORS[0], alpha=0.45)
    ax.set_title(dimension_names[dim])
    ax.set_ylim(-1.05, 1.05)
  for ax in axes[1]:
    ax.set_xlabel("time [s]")
  for ax in axes[:, 0]:
    ax.set_ylabel("action")
  fig.suptitle(f"Expert actions for {len(shown)} successful demos (normalized changes of joint targets)")
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def plot_phase_durations(demos, path):
  plt = plotting.plt
  successful = demos["success"]
  steps_per_phase = [(demos["phase"][successful] == phase).sum(axis=1) for phase in range(len(PHASE_NAMES))]
  fig, ax = plt.subplots(figsize=(7, 3.6))
  boxes = ax.boxplot(steps_per_phase, tick_labels=PHASE_NAMES, patch_artist=True, widths=0.5)
  for box in boxes["boxes"]:
    box.set(facecolor=plotting.COLORS[0], alpha=0.35, edgecolor=plotting.COLORS[0])
  for median_line in boxes["medians"]:
    median_line.set(color=plotting.TEXT_COLOR, linewidth=1.5)
  ax.set(ylabel="steps (x 0.02 s)", title="Time spent in each expert phase (successful demos)")
  fig.tight_layout()
  return plotting.save_figure(fig, path)


def render_videos_and_frames(demos, out_dir, video_dir, num_videos):
  from hw2.envs import load_pick_cube_env
  from hw2.render import TrajectoryRenderer, add_caption, save_video, tile_videos
  plt = plotting.plt

  env = load_pick_cube_env()
  successful_ids = np.flatnonzero(demos["success"])
  failed_ids = np.flatnonzero(~demos["success"])
  saved_videos = []

  renderer = TrajectoryRenderer(env.mj_model, height=480, width=640)
  for i in successful_ids[:num_videos]:
    frames = renderer.render_frames(demos["qpos"][i], demos["target_pos"][i], every_nth=2)
    frames = add_caption(frames, f"expert demo seed {demos['seeds'][i]} - success")
    saved_videos.append(str(save_video(video_dir / f"expert_demo_seed{demos['seeds'][i]}.mp4", frames, VIDEO_FPS)))
  for i in failed_ids[:1]:
    frames = renderer.render_frames(demos["qpos"][i], demos["target_pos"][i], every_nth=2)
    frames = add_caption(frames, f"expert seed {demos['seeds'][i]} - FAILURE")
    saved_videos.append(str(save_video(video_dir / f"expert_failure_seed{demos['seeds'][i]}.mp4", frames, VIDEO_FPS)))
  renderer.close()

  small_renderer = TrajectoryRenderer(env.mj_model, height=240, width=320)
  grid_videos = [add_caption(small_renderer.render_frames(demos["qpos"][i], demos["target_pos"][i], every_nth=2),
                             f"seed {demos['seeds'][i]}")
                 for i in successful_ids[:4]]
  saved_videos.append(str(save_video(video_dir / "expert_demos_grid.mp4", tile_videos(grid_videos, 2), VIDEO_FPS)))

  # Film strip: one row per demo, one column per time.
  frame_steps = [0, 25, 50, 75, 100, 150]
  rows = successful_ids[:4]
  fig, axes = plt.subplots(len(rows), len(frame_steps), figsize=(2.2 * len(frame_steps), 1.75 * len(rows)))
  for row, i in enumerate(rows):
    frames = small_renderer.render_frames(demos["qpos"][i][frame_steps], demos["target_pos"][i])
    for col, (frame, step) in enumerate(zip(frames, frame_steps)):
      ax = axes[row, col]
      ax.imshow(frame)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.grid(False)
      for spine in ax.spines.values():
        spine.set_visible(False)
      if row == 0:
        ax.set_title(f"t = {step * CONTROL_DT:.1f} s", fontsize=9)
      if col == 0:
        ax.set_ylabel(f"seed {demos['seeds'][i]}", fontsize=9)
  fig.suptitle("Expert demonstrations (red see-through box = target)", fontsize=11)
  fig.tight_layout()
  plotting.save_figure(fig, out_dir / "fig_frames.png")
  small_renderer.close()
  return saved_videos


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data", default="data/demos_baseline.npz")
  parser.add_argument("--out", default="results/demos")
  parser.add_argument("--video-dir", default="videos/expert")
  parser.add_argument("--num-videos", type=int, default=3)
  parser.add_argument("--no-render", action="store_true")
  args = parser.parse_args()

  plotting.use_report_style()
  out_dir = utils.repo_path(args.out)
  with np.load(utils.repo_path(args.data)) as saved:
    demos = {name: saved[name] for name in saved.files}

  stats = compute_dataset_stats(demos)
  utils.save_json(stats, out_dir / "demo_stats.json")
  plot_initial_conditions(demos, out_dir / "fig_coverage.png")
  plot_trajectories(demos, out_dir / "fig_trajectories.png")
  plot_actions(demos, out_dir / "fig_actions.png")
  plot_phase_durations(demos, out_dir / "fig_phases.png")
  videos = []
  if not args.no_render:
    videos = render_videos_and_frames(demos, out_dir, utils.repo_path(args.video_dir), args.num_videos)

  print("\n==== DEMO STATS (paste this back) ====")
  print(json.dumps({key: stats[key] for key in [
      "num_trajectories", "num_successful", "success_fraction", "episode_length_steps",
      "first_success_step_successful", "settle_step_successful", "gripper_path_length_m_successful",
      "final_dist_m_successful", "failure_modes", "action_saturation_fraction"]}, indent=2))
  print("figures ->", out_dir)
  for video in videos:
    print("video ->", video)


if __name__ == "__main__":
  main()
