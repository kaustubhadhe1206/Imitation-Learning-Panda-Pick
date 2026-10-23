"""Evaluate a trained student (or the scripted teacher) on held-out initial conditions.

Evaluation seeds are eval.seed_offset + [0, num_episodes), disjoint from the
demonstration seeds, and identical across all conditions. Each episode runs
the full 150-step horizon from env.reset(PRNGKey(seed)); nothing is replayed
from the training data.

Outputs in --out-dir:
  eval_summary.json   success rate (+ Wilson 95% CI), success vs threshold,
                      failure-mode counts, action smoothness, timing
  episodes.csv        one row per episode
Videos of successes / failures go to --video-dir. The raw rollout (.npz) is
kept next to the checkpoint (or in runs/teacher) for later re-rendering.

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
from hw2.rollout import episode_table, rollout, summarize


def run_eval(benv, controller, eval_cfg, seeds):
  B = eval_cfg["batch_size"]
  parts = []
  for s in range(0, len(seeds), B):
    batch = seeds[s:s + B]
    if len(batch) < B:  # keep the jit-compiled batch shape fixed
      batch = np.concatenate([batch, batch[-1] + 1 + np.arange(B - len(batch))])
    r = rollout(benv, batch, controller, benv.episode_length,
                success_threshold=eval_cfg["success_threshold"])
    parts.append(r)
  keys = [k for k in parts[0] if isinstance(parts[0][k], np.ndarray)]
  result = {k: np.concatenate([p[k] for p in parts])[:len(seeds)] for k in keys}
  return result


def write_videos(benv, result, rows, video_dir, label, num_videos):
  from hw2.render import TrajectoryRenderer, annotate, write_video
  renderer = TrajectoryRenderer(benv.mj_model, height=480, width=640)
  written = []
  succ = [i for i, r in enumerate(rows) if r["success"]]
  fail = [i for i, r in enumerate(rows) if not r["success"]]
  for kind, idx in [("success", succ[:num_videos]), ("failure", fail[:num_videos])]:
    for i in idx:
      seed = rows[i]["seed"]
      caption = f"{label} | eval seed {seed} | {rows[i]['failure_mode']} | final dist {rows[i]['final_dist'] * 100:.1f} cm"
      frames = annotate(renderer.render(result["qpos"][i], result["target_pos"][i], every=2), caption)
      written.append(str(write_video(Path(video_dir) / f"{kind}_seed{seed}.mp4", frames, fps=25)))
  renderer.close()
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--run-dir", help="trained student run directory (contains checkpoint.pt)")
  group.add_argument("--teacher", action="store_true", help="evaluate the scripted expert instead")
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--video-dir", default=None, help="default: videos/<out-dir name>")
  parser.add_argument("--num-videos", type=int, default=None)
  parser.add_argument("--collect-config", default="configs/collect.yaml",
                      help="teacher only: expert settings")
  parser.add_argument("--on-train-seeds", action="store_true",
                      help="sanity check only: evaluate on the run's own demo seeds (NOT held out)")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
  args = parser.parse_args()
  if args.on_train_seeds and args.teacher:
    parser.error("--on-train-seeds applies to student runs only")

  from hw2.envs import BatchedEnv
  out_dir = utils.resolve_path(args.out_dir)
  video_dir = utils.resolve_path(args.video_dir or f"videos/{out_dir.name}")
  benv = BatchedEnv()

  if args.teacher:
    from hw2.expert import ExpertController
    cfg = utils.load_config("configs/dp_base.yaml", args.overrides)
    collect_cfg = utils.load_config(args.collect_config)
    controller = ExpertController(benv.mj_model, benv.action_scale, collect_cfg["expert"])
    label, chunk = "teacher", cfg["policy"]["action_horizon"]
    raw_path = utils.resolve_path("runs/teacher/rollouts.npz")
  else:
    import torch
    from hw2.policy import DiffusionPolicyController, load_checkpoint
    run_dir = utils.resolve_path(args.run_dir)
    cfg = utils.load_config(run_dir / "config.yaml", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = load_checkpoint(run_dir / "checkpoint.pt", device)
    policy.num_inference_steps = cfg["policy"]["num_inference_steps"]
    controller = DiffusionPolicyController(policy, device, seed=cfg["eval"]["seed_offset"])
    label, chunk = f"student {run_dir.name}", policy.action_horizon
    raw_path = run_dir / "rollouts.npz"

  eval_cfg = cfg["eval"]
  seeds = eval_cfg["seed_offset"] + np.arange(eval_cfg["num_episodes"])
  if args.on_train_seeds:
    seeds = np.array(utils.load_json(run_dir / "train_summary.json")["demo_seeds"])
    label += " (TRAIN SEEDS)"
  t0 = time.time()
  result = run_eval(benv, controller, eval_cfg, seeds)
  wall = time.time() - t0

  rows = episode_table(result, chunk=chunk)
  summary = summarize(result, chunk=chunk)
  summary.update({
      "label": label,
      "eval_seeds": [int(result["seeds"][0]), int(result["seeds"][-1])],
      "wall_time_s": wall,
      "eval_config": eval_cfg,
      "env_info": utils.env_info(),
  })
  if not args.teacher:
    summary["num_policy_inferences"] = controller.num_inferences
    summary["num_inference_steps"] = cfg["policy"]["num_inference_steps"]

  out_dir.mkdir(parents=True, exist_ok=True)
  utils.save_json(summary, out_dir / "eval_summary.json")
  with open(out_dir / "episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  raw_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(raw_path, **{k: result[k] for k in [
      "seeds", "qpos", "target_pos", "actions", "dist", "success", "grip_pos", "box_pos"]})

  num_videos = eval_cfg.get("num_videos", 2) if args.num_videos is None else args.num_videos
  videos = []
  if num_videos > 0:
    try:
      videos = write_videos(benv, result, rows, video_dir, label, num_videos)
    except Exception as e:  # rendering problems must not lose the evaluation
      print(f"WARNING: video rendering failed: {e!r}")

  print("\n==== EVAL SUMMARY (paste this back) ====")
  print(json.dumps({k: summary[k] for k in [
      "label", "num_episodes", "success_rate", "success_ci95", "success_at_threshold",
      "final_dist_median", "failure_modes", "jerk_mean", "jerk_boundary", "jerk_within",
      "wall_time_s"] if k in summary}, indent=2))
  for v in videos:
    print("video ->", v)


if __name__ == "__main__":
  main()
