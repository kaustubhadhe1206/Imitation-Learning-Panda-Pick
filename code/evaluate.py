"""Evaluate a trained student policy (or the scripted teacher) on new, held-out starting positions (Part 4).

The evaluation seeds are eval.seed_offset + 0, 1, 2, ..., which never overlap
with the demo seeds, and are the same for every condition. Every episode is a
fresh env.reset(PRNGKey(seed)) run for the full 150 steps; nothing is replayed
from the training data.

Outputs in --out-dir:
  eval_summary.json   success rate (+ 95% CI), success at other thresholds,
                      failure-mode counts, action jerk, timing
  episodes.csv        one row per episode
Videos of some successes and failures go to --video-dir. The raw rollouts
(.npz) are saved next to the checkpoint (or in runs/teacher) so they can be
re-rendered later.

Usage:
  python code/evaluate.py --run-dir runs/n100_s0 --out-dir results/runs/n100_s0
  python code/evaluate.py --teacher --out-dir results/teacher_eval
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import hw2  # noqa: F401
from hw2 import utils
from hw2.rollout import make_episode_rows, run_episodes, summarize_episodes


def run_evaluation(batched_env, controller, eval_config, seeds):
  """Run all evaluation episodes in batches of eval_config["batch_size"]."""
  batch_size = eval_config["batch_size"]
  batches = []
  for start in range(0, len(seeds), batch_size):
    batch_seeds = seeds[start:start + batch_size]
    if len(batch_seeds) < batch_size:
      # Pad the last batch with extra seeds so the batch size (and the jit compile) stays the same.
      extra_seeds = batch_seeds[-1] + 1 + np.arange(batch_size - len(batch_seeds))
      batch_seeds = np.concatenate([batch_seeds, extra_seeds])
    batches.append(run_episodes(batched_env, batch_seeds, controller, batched_env.episode_length,
                                success_threshold=eval_config["success_threshold"]))
  array_names = [name for name in batches[0] if isinstance(batches[0][name], np.ndarray)]
  return {name: np.concatenate([batch[name] for batch in batches])[:len(seeds)] for name in array_names}


def save_example_videos(batched_env, results, episode_rows, video_dir, label, num_videos):
  from hw2.render import TrajectoryRenderer, add_caption, save_video
  renderer = TrajectoryRenderer(batched_env.mj_model, height=480, width=640)
  saved_videos = []
  success_ids = [i for i, row in enumerate(episode_rows) if row["success"]]
  failure_ids = [i for i, row in enumerate(episode_rows) if not row["success"]]
  for kind, episode_ids in [("success", success_ids[:num_videos]), ("failure", failure_ids[:num_videos])]:
    for i in episode_ids:
      row = episode_rows[i]
      caption = (f"{label} | eval seed {row['seed']} | {row['failure_mode']} | "
                 f"final dist {row['final_dist'] * 100:.1f} cm")
      frames = renderer.render_frames(results["qpos"][i], results["target_pos"][i], every_nth=2)
      frames = add_caption(frames, caption)
      saved_videos.append(str(save_video(Path(video_dir) / f"{kind}_seed{row['seed']}.mp4", frames, fps=25)))
  renderer.close()
  return saved_videos


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  which_policy = parser.add_mutually_exclusive_group(required=True)
  which_policy.add_argument("--run-dir", help="folder of a trained student (has checkpoint.pt)")
  which_policy.add_argument("--teacher", action="store_true", help="evaluate the scripted expert instead")
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--video-dir", default=None, help="default: videos/<name of out-dir>")
  parser.add_argument("--num-videos", type=int, default=None)
  parser.add_argument("--collect-config", default="configs/collect.yaml",
                      help="teacher only: where the expert settings are")
  parser.add_argument("--on-train-seeds", action="store_true",
                      help="sanity check only: evaluate on the run's own demo seeds (NOT held out)")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
  args = parser.parse_args()
  if args.on_train_seeds and args.teacher:
    parser.error("--on-train-seeds only works for student runs")

  from hw2.envs import BatchedPickCubeEnv
  out_dir = utils.repo_path(args.out_dir)
  video_dir = utils.repo_path(args.video_dir or f"videos/{out_dir.name}")
  batched_env = BatchedPickCubeEnv()

  if args.teacher:
    from hw2.expert import BatchedExpertController
    config = utils.load_config("configs/dp_base.yaml", args.overrides)
    collect_config = utils.load_config(args.collect_config)
    controller = BatchedExpertController(batched_env.mj_model, batched_env.action_scale, collect_config["expert"])
    label = "teacher"
    chunk_size = config["policy"]["action_horizon"]   # same split as the students, for comparison
    rollouts_path = utils.repo_path("runs/teacher/rollouts.npz")
  else:
    import torch
    from hw2.policy import DiffusionPolicyController, load_checkpoint
    run_dir = utils.repo_path(args.run_dir)
    config = utils.load_config(run_dir / "config.yaml", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = load_checkpoint(run_dir / "checkpoint.pt", device)
    policy.num_inference_steps = config["policy"]["num_inference_steps"]
    controller = DiffusionPolicyController(policy, device, seed=config["eval"]["seed_offset"])
    label = f"student {run_dir.name}"
    chunk_size = policy.action_horizon
    rollouts_path = run_dir / "rollouts.npz"

  eval_config = config["eval"]
  seeds = eval_config["seed_offset"] + np.arange(eval_config["num_episodes"])
  if args.on_train_seeds:
    seeds = np.array(utils.load_json(run_dir / "train_summary.json")["demo_seeds"])
    label += " (TRAIN SEEDS)"
  start_time = time.time()
  results = run_evaluation(batched_env, controller, eval_config, seeds)
  eval_seconds = time.time() - start_time

  episode_rows = make_episode_rows(results, chunk_size=chunk_size)
  summary = summarize_episodes(results, chunk_size=chunk_size)
  summary.update({
      "label": label,
      "eval_seeds": [int(results["seeds"][0]), int(results["seeds"][-1])],
      "wall_time_s": eval_seconds,
      "eval_config": eval_config,
      "system_info": utils.get_system_info(),
  })
  if not args.teacher:
    summary["num_policy_calls"] = controller.num_policy_calls
    summary["num_inference_steps"] = config["policy"]["num_inference_steps"]

  out_dir.mkdir(parents=True, exist_ok=True)
  utils.save_json(summary, out_dir / "eval_summary.json")
  with open(out_dir / "episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
    writer.writeheader()
    writer.writerows(episode_rows)
  rollouts_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(rollouts_path, **{name: results[name] for name in [
      "seeds", "qpos", "target_pos", "actions", "dist", "success", "grip_pos", "box_pos"]})

  num_videos = eval_config.get("num_videos", 2) if args.num_videos is None else args.num_videos
  videos = []
  if num_videos > 0:
    try:
      videos = save_example_videos(batched_env, results, episode_rows, video_dir, label, num_videos)
    except Exception as error:   # a rendering problem shouldn't throw away the evaluation results
      print(f"WARNING: video rendering failed: {error!r}")

  print("\n==== EVAL SUMMARY (paste this back) ====")
  print(json.dumps({key: summary[key] for key in [
      "label", "num_episodes", "success_rate", "success_ci95", "success_at_threshold",
      "final_dist_median", "failure_modes", "jerk_mean", "jerk_boundary", "jerk_within",
      "wall_time_s"] if key in summary}, indent=2))
  for video in videos:
    print("video ->", video)


if __name__ == "__main__":
  main()
