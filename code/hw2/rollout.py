"""Backend-agnostic batched rollouts, success labels and failure diagnostics.

A *controller* is any object with
  reset(snap)          called once with the initial batched snapshot
  act(snap, t) -> (B, action_dim) array in [-1, 1]
and optionally a `phase` attribute (int array, B) that is recorded per step.

A *batched env* provides reset(seeds) -> state, step(state, actions) -> state
and snapshot(state) -> dict of numpy arrays (see envs.BatchedEnv).

Success is NOT defined by MuJoCo Playground for PandaPickCube, so we define it
here (and fix it before running any experiment): the box centre is within
`success_threshold` metres of the target at the final step and the box never
left the workspace (the env's out-of-bounds `done` flag).
"""

from collections import defaultdict

import numpy as np

SUCCESS_THRESHOLD = 0.05
TABLE_BOX_Z = 0.03          # box centre height when resting on the floor
LIFTED_DZ = 0.02            # box counts as lifted when > 2 cm above rest
REACH_TOL = 0.02            # gripper "reached" the box if within 2 cm

FAILURE_MODES = ["success", "out_of_bounds", "never_reached", "grasp_failed",
                 "dropped", "inaccurate_place", "late"]


def rollout(benv, seeds, controller, num_steps, noise_std=0.0, noise_rng=None,
            success_threshold=SUCCESS_THRESHOLD):
  """Run one batched episode per seed. Returns a dict of numpy arrays.

  Per-step arrays have shape (B, T, ...) where entry t holds the observation
  the controller saw at step t and the action it executed at step t.
  Positional arrays (qpos, grip_pos, box_pos, dist) have T + 1 entries
  (including the state after the last action), for rendering and labels.

  If noise_std > 0, i.i.d. Gaussian noise is added to the controller's action
  *before* it is executed, then clipped to [-1, 1]; `actions` stores what was
  actually executed and `actions_clean` what the controller intended.
  """
  seeds = np.asarray(seeds)
  if noise_std > 0 and noise_rng is None:
    raise ValueError("noise_rng is required when noise_std > 0")

  state = benv.reset(seeds)
  snap = benv.snapshot(state)
  controller.reset(snap)

  rec = defaultdict(list)

  def record_positions(s):
    for k in ("qpos", "grip_pos", "box_pos"):
      rec[k].append(s[k])

  record_positions(snap)
  target_pos = snap["target_pos"].copy()
  for t in range(num_steps):
    a_clean = np.clip(np.asarray(controller.act(snap, t), dtype=np.float32), -1.0, 1.0)
    a = a_clean
    if noise_std > 0:
      a = np.clip(a + noise_rng.normal(0.0, noise_std, a.shape), -1.0, 1.0).astype(np.float32)
    rec["obs"].append(snap["obs"])
    rec["actions"].append(a)
    rec["actions_clean"].append(a_clean)
    if hasattr(controller, "phase"):
      rec["phase"].append(np.asarray(controller.phase, dtype=np.int8))

    state = benv.step(state, a)
    snap = benv.snapshot(state)
    rec["rewards"].append(snap["reward"])
    rec["dones"].append(snap["done"])
    rec["reached_box"].append(snap["reached_box"])
    record_positions(snap)

  out = {k: np.stack(v, axis=1) for k, v in rec.items()}
  out["seeds"] = seeds
  out["target_pos"] = target_pos
  out["box_init"] = out["box_pos"][:, 0].copy()
  out.update(success_labels(out["box_pos"], target_pos, out["dones"], success_threshold))
  return out


def success_labels(box_pos, target_pos, dones, success_threshold=SUCCESS_THRESHOLD):
  dist = np.linalg.norm(box_pos - target_pos[:, None, :], axis=-1)        # (B, T+1)
  oob_step = np.concatenate([np.zeros_like(dones[:, :1]), dones], axis=1) > 0.5
  oob_so_far = np.maximum.accumulate(oob_step, axis=1)
  success_step = (dist < success_threshold) & ~oob_so_far
  any_success = success_step.any(axis=1)
  first = np.where(any_success, success_step.argmax(axis=1), -1)
  return {
      "dist": dist.astype(np.float32),
      "success_step": success_step,
      "success": success_step[:, -1],
      "first_success_step": first.astype(np.int32),
      "out_of_bounds": oob_so_far[:, -1],
  }


def classify_failures(result):
  """One label per episode from FAILURE_MODES (first matching rule wins)."""
  grip_box = np.linalg.norm(result["grip_pos"] - result["box_pos"], axis=-1)
  box_z = result["box_pos"][..., 2]
  lifted_any = (box_z > TABLE_BOX_Z + LIFTED_DZ).any(axis=1)
  lifted_end = box_z[:, -1] > TABLE_BOX_Z + LIFTED_DZ
  modes = []
  for i in range(len(result["success"])):
    if result["success"][i]:
      modes.append("success")
    elif result["out_of_bounds"][i]:
      modes.append("out_of_bounds")
    elif grip_box[i].min() > REACH_TOL:
      modes.append("never_reached")
    elif not lifted_any[i]:
      modes.append("grasp_failed")
    elif not lifted_end[i]:
      modes.append("dropped")
    elif result["success_step"][i].any():
      modes.append("late")          # was within threshold but drifted away by the end
    else:
      modes.append("inaccurate_place")
  return np.array(modes)


def action_smoothness(actions, chunk=None):
  """Mean L2 action change per step; optionally split at chunk boundaries.

  For a receding-horizon policy that executes `chunk` actions per inference,
  boundary steps are t = k * chunk (k >= 1), where a new chunk starts.
  """
  diffs = np.linalg.norm(np.diff(actions, axis=1), axis=-1)      # (B, T-1), diff[t-1] = |a_t - a_{t-1}|
  out = {"jerk_mean": diffs.mean(axis=1)}
  if chunk:
    t = np.arange(1, actions.shape[1])
    boundary = (t % chunk) == 0
    out["jerk_boundary"] = diffs[:, boundary].mean(axis=1)
    out["jerk_within"] = diffs[:, ~boundary].mean(axis=1)
  return out


def episode_table(result, chunk=None):
  """Per-episode diagnostics as a list of dicts (one CSV row each)."""
  modes = classify_failures(result)
  smooth = action_smoothness(result["actions"], chunk)
  grip_box = np.linalg.norm(result["grip_pos"] - result["box_pos"], axis=-1)
  rows = []
  for i in range(len(result["seeds"])):
    row = {
        "seed": int(result["seeds"][i]),
        "success": bool(result["success"][i]),
        "failure_mode": str(modes[i]),
        "final_dist": float(result["dist"][i, -1]),
        "min_dist": float(result["dist"][i].min()),
        "first_success_step": int(result["first_success_step"][i]),
        "out_of_bounds": bool(result["out_of_bounds"][i]),
        "min_grip_box_dist": float(grip_box[i].min()),
        "max_box_z": float(result["box_pos"][i, :, 2].max()),
        "reached_box": bool(result["reached_box"][i, -1] > 0.5),
        "return": float(result["rewards"][i].sum()),
    }
    for k, v in smooth.items():
      row[k] = float(v[i])
    rows.append(row)
  return rows


def summarize(result, thresholds=(0.02, 0.03, 0.05, 0.10), chunk=None):
  from .utils import wilson_interval
  n = len(result["seeds"])
  k = int(result["success"].sum())
  lo, hi = wilson_interval(k, n)
  modes = classify_failures(result)
  smooth = action_smoothness(result["actions"], chunk)
  final = result["dist"][:, -1]
  summary = {
      "num_episodes": n,
      "num_success": k,
      "success_rate": k / n,
      "success_ci95": [lo, hi],
      "success_at_threshold": {
          f"{th:.2f}": float(((final < th) & ~result["out_of_bounds"]).mean()) for th in thresholds},
      "final_dist_mean": float(final.mean()),
      "final_dist_median": float(np.median(final)),
      "failure_modes": {m: int((modes == m).sum()) for m in FAILURE_MODES},
      "first_success_step_mean": (float(result["first_success_step"][result["first_success_step"] >= 0].mean())
                                  if (result["first_success_step"] >= 0).any() else None),
  }
  for key, v in smooth.items():
    summary[key] = float(v.mean())
  return summary


class ReplayController:
  """Open-loop replay of recorded actions (used to verify action/obs alignment)."""

  def __init__(self, actions):
    self.actions = np.asarray(actions)

  def reset(self, snap):
    pass

  def act(self, snap, t):
    return self.actions[:, t]
