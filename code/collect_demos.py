"""Collect demonstrations from the scripted expert (Part 1).

The expert runs on seeds seed_start, seed_start + 1, ... in parallel batches
until we have `num_success` successful episodes. We keep every attempt up to
(and including) the one that gave the last needed success, with its
success/failure label. Training only uses the successful ones, but keeping the
failures lets us report the expert's real success rate.

Usage:
  python code/collect_demos.py                                  # baseline dataset
  python code/collect_demos.py --set noise_std=0.3 name=noise03 \
      out=data/demos_noise03.npz summary_out=results/demos/collection_noise03.json
"""

import argparse
import json
import time

import numpy as np

import hw2  # noqa: F401  (sets MUJOCO_GL / XLA env variables before jax and mujoco get imported)
from hw2 import utils
from hw2.rollout import ReplayActionsController, label_failure_modes, run_episodes

# Which result arrays get saved, and which of those are stored as float32.
FLOAT_ARRAYS = ["obs", "actions", "actions_clean", "rewards", "dones", "reached_box", "qpos",
                "grip_pos", "box_pos", "target_pos", "box_init", "dist"]
OTHER_ARRAYS = ["seeds", "phase", "success_step", "success", "first_success_step", "out_of_bounds"]


def count_steps_per_phase(phases, num_phases):
  """How many steps each episode spent in each expert phase -> (num_episodes, num_phases)."""
  return np.stack([(phases == phase).sum(axis=1) for phase in range(num_phases)], axis=1)


def check_replay_matches(batched_env, batch_results, num_steps):
  """Replay the saved actions open-loop from the same seeds; the observations must come out the same.

  This checks that every stored action really belongs to the observation stored next to it.
  """
  replay = run_episodes(batched_env, batch_results["seeds"],
                        ReplayActionsController(batch_results["actions"]), num_steps)
  obs_difference = np.abs(replay["obs"] - batch_results["obs"]).max(axis=(1, 2))
  return {
      "episodes": int(len(obs_difference)),
      "max_abs_obs_diff": float(obs_difference.max()),
      "median_abs_obs_diff": float(np.median(obs_difference)),
      "success_agreement": float((replay["success"] == batch_results["success"]).mean()),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  utils.add_config_arguments(parser, "configs/collect.yaml")
  args = parser.parse_args()
  config = utils.load_config(args.config, args.overrides)
  print(json.dumps(config, indent=2))

  from hw2.envs import BatchedPickCubeEnv, get_env_config
  from hw2.expert import DEFAULT_EXPERT_CONFIG, PHASE_NAMES, BatchedExpertController

  batched_env = BatchedPickCubeEnv()
  expert = BatchedExpertController(batched_env.mj_model, batched_env.action_scale, config["expert"])
  noise_rng = np.random.default_rng(config["noise_seed"])
  batch_size = config["batch_size"]
  num_steps = batched_env.episode_length

  batches = []
  total_successes = 0
  next_seed = config["seed_start"]
  start_time = time.time()
  while (total_successes < config["num_success"]
         and next_seed - config["seed_start"] < config["max_attempts"]):
    seeds = np.arange(next_seed, next_seed + batch_size)
    batch = run_episodes(batched_env, seeds, expert, num_steps, noise_std=config["noise_std"],
                         noise_rng=noise_rng, success_threshold=config["success_threshold"])
    batches.append(batch)
    total_successes += int(batch["success"].sum())
    next_seed += batch_size
    print(f"seeds {seeds[0]}-{seeds[-1]}: {int(batch['success'].sum())}/{batch_size} succeeded "
          f"(total {total_successes}) [{time.time() - start_time:.0f}s]", flush=True)

  demos = {name: np.concatenate([batch[name] for batch in batches]) for name in FLOAT_ARRAYS + OTHER_ARRAYS}
  # Keep attempts up to the one that gave the num_success-th success (a continuous range of seeds).
  successes_so_far = np.cumsum(demos["success"])
  if successes_so_far[-1] >= config["num_success"]:
    num_to_keep = int(np.searchsorted(successes_so_far, config["num_success"]) + 1)
  else:
    num_to_keep = len(successes_so_far)
    print(f"WARNING: only {int(successes_so_far[-1])} successes in max_attempts={config['max_attempts']}")
  demos = {name: values[:num_to_keep] for name, values in demos.items()}

  replay_check = None
  if config["verify_replay"]:
    replay_check = check_replay_matches(batched_env, batches[0], num_steps)
    print("replay check:", replay_check)

  failure_modes = label_failure_modes(demos)
  successful = demos["success"]
  steps_per_phase = count_steps_per_phase(demos["phase"], len(PHASE_NAMES))
  first_success_steps = demos["first_success_step"][successful]
  went_back_to_earlier_phase = (np.diff(demos["phase"].astype(int), axis=1) < 0).any(axis=1)
  summary = {
      "name": config["name"],
      "num_attempts": int(len(successful)),
      "num_success": int(successful.sum()),
      "teacher_success_rate": float(successful.mean()),
      "seed_range": [int(demos["seeds"][0]), int(demos["seeds"][-1])],
      "episode_length": num_steps,
      "noise_std": config["noise_std"],
      "first_success_step": ({"mean": float(first_success_steps.mean()), "std": float(first_success_steps.std())}
                             if len(first_success_steps) else None),
      "final_dist_success_mean": float(demos["dist"][successful, -1].mean()) if successful.any() else None,
      "failure_modes": {mode: int((failure_modes == mode).sum()) for mode in np.unique(failure_modes)},
      "phase_steps_mean_success": {name: float(steps_per_phase[successful, i].mean())
                                   for i, name in enumerate(PHASE_NAMES)},
      "regrasp_episodes": int(went_back_to_earlier_phase.sum()),
      "replay_check": replay_check,
      "wall_time_s": time.time() - start_time,
      "expert_config": {**DEFAULT_EXPERT_CONFIG, **config["expert"]},
      "env_config": get_env_config(),
      "config": config,
      "system_info": utils.get_system_info(),
  }
  out_path = utils.repo_path(config["out"])
  out_path.parent.mkdir(parents=True, exist_ok=True)
  arrays_to_save = {name: (values.astype(np.float32) if name in FLOAT_ARRAYS else values)
                    for name, values in demos.items()}
  np.savez_compressed(out_path, meta=np.array(json.dumps(summary, default=str)), **arrays_to_save)
  utils.save_json(summary, utils.repo_path(config["summary_out"]))

  print("\n==== COLLECTION SUMMARY (paste this back) ====")
  print(json.dumps({key: summary[key] for key in [
      "num_attempts", "num_success", "teacher_success_rate", "first_success_step",
      "final_dist_success_mean", "failure_modes", "phase_steps_mean_success",
      "regrasp_episodes", "replay_check", "wall_time_s"]}, indent=2))
  print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
