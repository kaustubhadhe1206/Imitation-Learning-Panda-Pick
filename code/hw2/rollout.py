"""Running episodes, labelling success/failure, and computing per-episode stats.

A "controller" is anything with two methods:
  reset(env_state)               called once at the start with the batched state
  get_action(env_state, step)    returns a (batch_size, 8) array of actions in [-1, 1]
It can also have a `current_phases` attribute (the expert does), which gets recorded.

The batched env needs reset(seeds), step(state, actions) and to_numpy(state)
(see envs.BatchedPickCubeEnv). Tests use a fake env with the same methods.

How we define success (PandaPickCube doesn't have its own success flag):
the box center is within SUCCESS_THRESHOLD meters of the target at the LAST
step, and the box never went out of bounds (the env's `done` flag).
"""

from collections import defaultdict

import numpy as np

SUCCESS_THRESHOLD = 0.05   # meters
BOX_REST_HEIGHT = 0.03     # height of the box center when it sits on the table
LIFTED_MARGIN = 0.02       # box counts as "lifted" when it is 2 cm above its rest height
REACHED_DISTANCE = 0.02    # gripper counts as having "reached" the box within 2 cm

FAILURE_MODES = ["success", "out_of_bounds", "never_reached", "grasp_failed",
                 "dropped", "inaccurate_place", "late"]


def run_episodes(batched_env, seeds, controller, num_steps, noise_std=0.0, noise_rng=None,
                 success_threshold=SUCCESS_THRESHOLD):
  """Run one episode per seed (all in parallel) and return everything as numpy arrays.

  Arrays per step have shape (num_episodes, num_steps, ...): entry t is the
  observation the controller saw at step t and the action it took at step t.
  Position arrays (qpos, grip_pos, box_pos, dist) have num_steps + 1 entries,
  because they also include the state after the last action.

  If noise_std > 0, Gaussian noise is added to each action before it is sent
  to the env (then clipped to [-1, 1]). "actions" is what was really executed,
  "actions_clean" is what the controller wanted.
  """
  seeds = np.asarray(seeds)
  if noise_std > 0 and noise_rng is None:
    raise ValueError("noise_rng is needed when noise_std > 0")

  sim_state = batched_env.reset(seeds)
  env_state = batched_env.to_numpy(sim_state)
  controller.reset(env_state)

  recorded = defaultdict(list)

  def record_positions(state):
    for name in ("qpos", "grip_pos", "box_pos"):
      recorded[name].append(state[name])

  record_positions(env_state)
  target_positions = env_state["target_pos"].copy()
  for step in range(num_steps):
    wanted_action = np.clip(np.asarray(controller.get_action(env_state, step), dtype=np.float32), -1.0, 1.0)
    executed_action = wanted_action
    if noise_std > 0:
      noise = noise_rng.normal(0.0, noise_std, wanted_action.shape)
      executed_action = np.clip(wanted_action + noise, -1.0, 1.0).astype(np.float32)

    recorded["obs"].append(env_state["obs"])
    recorded["actions"].append(executed_action)
    recorded["actions_clean"].append(wanted_action)
    if hasattr(controller, "current_phases"):
      recorded["phase"].append(np.asarray(controller.current_phases, dtype=np.int8))

    sim_state = batched_env.step(sim_state, executed_action)
    env_state = batched_env.to_numpy(sim_state)
    recorded["rewards"].append(env_state["reward"])
    recorded["dones"].append(env_state["done"])
    recorded["reached_box"].append(env_state["reached_box"])
    record_positions(env_state)

  results = {name: np.stack(values, axis=1) for name, values in recorded.items()}
  results["seeds"] = seeds
  results["target_pos"] = target_positions
  results["box_init"] = results["box_pos"][:, 0].copy()
  results.update(compute_success_labels(results["box_pos"], target_positions, results["dones"],
                                        success_threshold))
  return results


def compute_success_labels(box_positions, target_positions, done_flags, success_threshold=SUCCESS_THRESHOLD):
  """Distance to target at every step, and the success flags that follow from it."""
  distance_to_target = np.linalg.norm(box_positions - target_positions[:, None, :], axis=-1)   # (N, T+1)
  # done_flags[t] is set after action t, so shift by one to line up with positions.
  out_of_bounds_now = np.concatenate([np.zeros_like(done_flags[:, :1]), done_flags], axis=1) > 0.5
  out_of_bounds_so_far = np.maximum.accumulate(out_of_bounds_now, axis=1)
  success_at_step = (distance_to_target < success_threshold) & ~out_of_bounds_so_far
  ever_succeeded = success_at_step.any(axis=1)
  first_success_step = np.where(ever_succeeded, success_at_step.argmax(axis=1), -1)
  return {
      "dist": distance_to_target.astype(np.float32),
      "success_step": success_at_step,
      "success": success_at_step[:, -1],
      "first_success_step": first_success_step.astype(np.int32),
      "out_of_bounds": out_of_bounds_so_far[:, -1],
  }


def label_failure_modes(results):
  """Give each episode one label from FAILURE_MODES (the first rule that matches wins)."""
  gripper_to_box = np.linalg.norm(results["grip_pos"] - results["box_pos"], axis=-1)
  box_height = results["box_pos"][..., 2]
  was_lifted = (box_height > BOX_REST_HEIGHT + LIFTED_MARGIN).any(axis=1)
  lifted_at_end = box_height[:, -1] > BOX_REST_HEIGHT + LIFTED_MARGIN
  labels = []
  for i in range(len(results["success"])):
    if results["success"][i]:
      labels.append("success")
    elif results["out_of_bounds"][i]:
      labels.append("out_of_bounds")
    elif gripper_to_box[i].min() > REACHED_DISTANCE:
      labels.append("never_reached")
    elif not was_lifted[i]:
      labels.append("grasp_failed")
    elif not lifted_at_end[i]:
      labels.append("dropped")
    elif results["success_step"][i].any():
      labels.append("late")            # was at the target at some point but drifted away
    else:
      labels.append("inaccurate_place")
  return np.array(labels)


def compute_action_jerk(actions, chunk_size=None):
  """How much the action changes from one step to the next (mean of |a_t - a_{t-1}|).

  For a policy that plans `chunk_size` actions at a time, we also split this
  into steps where a new chunk starts (t = chunk_size, 2*chunk_size, ...) and
  steps inside a chunk, to see if the policy "jumps" between chunks.
  """
  step_changes = np.linalg.norm(np.diff(actions, axis=1), axis=-1)   # entry t-1 = |a_t - a_{t-1}|
  jerk = {"jerk_mean": step_changes.mean(axis=1)}
  if chunk_size:
    step_numbers = np.arange(1, actions.shape[1])
    is_chunk_start = (step_numbers % chunk_size) == 0
    jerk["jerk_boundary"] = step_changes[:, is_chunk_start].mean(axis=1)
    jerk["jerk_within"] = step_changes[:, ~is_chunk_start].mean(axis=1)
  return jerk


def make_episode_rows(results, chunk_size=None):
  """One dict per episode with its diagnostics (these become rows of episodes.csv)."""
  failure_modes = label_failure_modes(results)
  jerk = compute_action_jerk(results["actions"], chunk_size)
  gripper_to_box = np.linalg.norm(results["grip_pos"] - results["box_pos"], axis=-1)
  rows = []
  for i in range(len(results["seeds"])):
    row = {
        "seed": int(results["seeds"][i]),
        "success": bool(results["success"][i]),
        "failure_mode": str(failure_modes[i]),
        "final_dist": float(results["dist"][i, -1]),
        "min_dist": float(results["dist"][i].min()),
        "first_success_step": int(results["first_success_step"][i]),
        "out_of_bounds": bool(results["out_of_bounds"][i]),
        "min_grip_box_dist": float(gripper_to_box[i].min()),
        "max_box_z": float(results["box_pos"][i, :, 2].max()),
        "reached_box": bool(results["reached_box"][i, -1] > 0.5),
        "return": float(results["rewards"][i].sum()),
    }
    for name, values in jerk.items():
      row[name] = float(values[i])
    rows.append(row)
  return rows


def summarize_episodes(results, thresholds=(0.02, 0.03, 0.05, 0.10), chunk_size=None):
  """Success rate (with 95% CI), success at other thresholds, failure counts, and jerk."""
  from .utils import wilson_confidence_interval
  num_episodes = len(results["seeds"])
  num_success = int(results["success"].sum())
  ci_low, ci_high = wilson_confidence_interval(num_success, num_episodes)
  failure_modes = label_failure_modes(results)
  jerk = compute_action_jerk(results["actions"], chunk_size)
  final_distance = results["dist"][:, -1]
  first_success = results["first_success_step"]
  summary = {
      "num_episodes": num_episodes,
      "num_success": num_success,
      "success_rate": num_success / num_episodes,
      "success_ci95": [ci_low, ci_high],
      "success_at_threshold": {
          f"{threshold:.2f}": float(((final_distance < threshold) & ~results["out_of_bounds"]).mean())
          for threshold in thresholds},
      "final_dist_mean": float(final_distance.mean()),
      "final_dist_median": float(np.median(final_distance)),
      "failure_modes": {mode: int((failure_modes == mode).sum()) for mode in FAILURE_MODES},
      "first_success_step_mean": (float(first_success[first_success >= 0].mean())
                                  if (first_success >= 0).any() else None),
  }
  for name, values in jerk.items():
    summary[name] = float(values.mean())
  return summary


class ReplayActionsController:
  """Plays back recorded actions without looking at the state (used to check that
  stored actions really line up with stored observations)."""

  def __init__(self, actions):
    self.actions = np.asarray(actions)

  def reset(self, env_state):
    pass

  def get_action(self, env_state, step):
    return self.actions[:, step]
