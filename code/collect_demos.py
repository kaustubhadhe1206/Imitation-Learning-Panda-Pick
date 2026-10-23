"""Collect scripted-expert demonstrations for PandaPickCube.

Runs the expert on demo seeds seed_start, seed_start + 1, ... in parallel
batches until `num_success` episodes succeed, then keeps the contiguous seed
prefix ending at the num_success-th success. *All* attempts in that prefix are
stored with success labels (training filters to successes), so the teacher's
raw success rate is recoverable from the file.

Usage:
  python code/collect_demos.py                                  # baseline
  python code/collect_demos.py --set noise_std=0.3 name=noise03 \
      out=data/demos_noise03.npz summary_out=results/demos/collection_noise03.json
"""

import argparse
import json
import time

import numpy as np

import hw2  # noqa: F401  (sets MUJOCO_GL / XLA env vars before jax/mujoco import)
from hw2 import utils
from hw2.rollout import ReplayController, classify_failures, rollout

SAVE_FLOAT = ["obs", "actions", "actions_clean", "rewards", "dones", "reached_box", "qpos",
              "grip_pos", "box_pos", "target_pos", "box_init", "dist"]
SAVE_OTHER = ["seeds", "phase", "success_step", "success", "first_success_step", "out_of_bounds"]


def phase_durations(phase, num_phases):
  """Steps spent in each phase per episode -> (N, num_phases)."""
  return np.stack([(phase == p).sum(axis=1) for p in range(num_phases)], axis=1)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  utils.add_config_args(parser, "configs/collect.yaml")
  args = parser.parse_args()
  cfg = utils.load_config(args.config, args.overrides)
  print(json.dumps(cfg, indent=2))

  from hw2.envs import BatchedEnv, env_config_dict
  from hw2.expert import DEFAULT_EXPERT_CFG, PHASE_NAMES, ExpertController

  benv = BatchedEnv()
  expert = ExpertController(benv.mj_model, benv.action_scale, cfg["expert"])
  noise_rng = np.random.default_rng(cfg["noise_seed"])
  B, T = cfg["batch_size"], benv.episode_length

  batches, n_success, next_seed = [], 0, cfg["seed_start"]
  t0 = time.time()
  while n_success < cfg["num_success"] and next_seed - cfg["seed_start"] < cfg["max_attempts"]:
    seeds = np.arange(next_seed, next_seed + B)
    r = rollout(benv, seeds, expert, T, noise_std=cfg["noise_std"], noise_rng=noise_rng,
                success_threshold=cfg["success_threshold"])
    batches.append(r)
    n_success += int(r["success"].sum())
    next_seed += B
    print(f"seeds {seeds[0]}-{seeds[-1]}: {int(r['success'].sum())}/{B} succeeded "
          f"(total {n_success}) [{time.time() - t0:.0f}s]", flush=True)

  data = {k: np.concatenate([r[k] for r in batches]) for k in SAVE_FLOAT + SAVE_OTHER}
  cum = np.cumsum(data["success"])
  keep = int(np.searchsorted(cum, cfg["num_success"]) + 1) if cum[-1] >= cfg["num_success"] else len(cum)
  data = {k: v[:keep] for k, v in data.items()}
  if cum[-1] < cfg["num_success"]:
    print(f"WARNING: only {int(cum[-1])} successes within max_attempts={cfg['max_attempts']}")

  # Action/observation alignment check: replaying the stored (executed)
  # actions open-loop from the same seeds must reproduce the stored observations.
  replay = None
  if cfg["verify_replay"]:
    first = batches[0]
    rr = rollout(benv, first["seeds"], ReplayController(first["actions"]), T)
    err = np.abs(rr["obs"] - first["obs"]).max(axis=(1, 2))
    replay = {"episodes": int(len(err)), "max_abs_obs_diff": float(err.max()),
              "median_abs_obs_diff": float(np.median(err)),
              "success_agreement": float((rr["success"] == first["success"]).mean())}
    print("replay check:", replay)

  modes = classify_failures(data)
  succ = data["success"]
  durations = phase_durations(data["phase"], len(PHASE_NAMES))
  first = data["first_success_step"][succ]
  summary = {
      "name": cfg["name"],
      "num_attempts": int(len(succ)),
      "num_success": int(succ.sum()),
      "teacher_success_rate": float(succ.mean()),
      "seed_range": [int(data["seeds"][0]), int(data["seeds"][-1])],
      "episode_length": T,
      "noise_std": cfg["noise_std"],
      "first_success_step": {"mean": float(first.mean()), "std": float(first.std())} if len(first) else None,
      "final_dist_success_mean": float(data["dist"][succ, -1].mean()) if succ.any() else None,
      "failure_modes": {m: int((modes == m).sum()) for m in np.unique(modes)},
      "phase_steps_mean_success": {n: float(durations[succ, i].mean()) for i, n in enumerate(PHASE_NAMES)},
      "regrasp_episodes": int(((np.diff(data["phase"].astype(int), axis=1) < 0).any(axis=1)).sum()),
      "replay_check": replay,
      "wall_time_s": time.time() - t0,
      "expert_cfg": {**DEFAULT_EXPERT_CFG, **cfg["expert"]},
      "env_config": env_config_dict(),
      "config": cfg,
      "env_info": utils.env_info(),
  }
  out = utils.resolve_path(cfg["out"])
  out.parent.mkdir(parents=True, exist_ok=True)
  arrays = {k: (v.astype(np.float32) if k in SAVE_FLOAT else v) for k, v in data.items()}
  np.savez_compressed(out, meta=np.array(json.dumps(summary, default=str)), **arrays)
  utils.save_json(summary, utils.resolve_path(cfg["summary_out"]))

  print("\n==== COLLECTION SUMMARY (paste this back) ====")
  print(json.dumps({k: summary[k] for k in [
      "num_attempts", "num_success", "teacher_success_rate", "first_success_step",
      "final_dist_success_mean", "failure_modes", "phase_steps_mean_success",
      "regrasp_episodes", "replay_check", "wall_time_s"]}, indent=2))
  print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
