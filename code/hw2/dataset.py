"""Demonstration loading, normalization and sequence windows for Diffusion Policy.

Windowing follows Chi et al.'s state-based Diffusion Policy notebook: every
training sample is a length-`pred_horizon` action sequence starting at time
t, conditioned on the `obs_horizon` observations starting at t. Episodes are
edge-padded (first frame repeated `obs_horizon - 1` times before the start,
last frame repeated `action_horizon - 1` times after the end), which here is
implemented as index clipping.
"""

import numpy as np


def load_demos(path, num_demos=None, success_only=True):
  """Load demonstrations; subsets are the first `num_demos` successes by seed.

  Taking a prefix in seed order makes the 25-demo set a subset of the
  50-demo set, which is a subset of the 100-demo set.
  """
  with np.load(path) as z:
    idx = np.flatnonzero(z["success"]) if success_only else np.arange(len(z["seeds"]))
    idx = idx[np.argsort(z["seeds"][idx], kind="stable")]
    if num_demos is not None:
      if len(idx) < num_demos:
        raise ValueError(f"Requested {num_demos} demos but {path} has only {len(idx)}")
      idx = idx[:num_demos]
    return {
        "obs": z["obs"][idx].astype(np.float32),
        "actions": z["actions"][idx].astype(np.float32),
        "seeds": z["seeds"][idx],
    }


class MinMaxNormalizer:
  """Per-dimension affine map of the data range to [-1, 1]."""

  def __init__(self, data_min, data_max, eps=1e-4):
    self.min = np.asarray(data_min, dtype=np.float32)
    self.max = np.asarray(data_max, dtype=np.float32)
    rng = self.max - self.min
    # Near-constant dimensions: map to ~0 instead of dividing by ~0.
    self.scale = np.where(rng > eps, 2.0 / np.maximum(rng, eps), 1.0).astype(np.float32)
    self.offset = np.where(rng > eps, -1.0 - self.min * self.scale, -self.min).astype(np.float32)

  @classmethod
  def fit(cls, x):
    x = x.reshape(-1, x.shape[-1])
    return cls(x.min(axis=0), x.max(axis=0))

  def normalize(self, x):
    return x * self.scale + self.offset

  def unnormalize(self, y):
    return (y - self.offset) / self.scale

  def state(self):
    return {"min": self.min, "max": self.max}


def window_indices(length, obs_horizon, pred_horizon, action_horizon):
  """(num_windows, pred_horizon) time indices into one episode, edge-clipped."""
  starts = np.arange(-(obs_horizon - 1), length - pred_horizon + action_horizon)
  idx = starts[:, None] + np.arange(pred_horizon)[None, :]
  return np.clip(idx, 0, length - 1)


def make_windows(obs, actions, obs_horizon, pred_horizon, action_horizon):
  """Stack all windows of all episodes.

  obs: (N, T, obs_dim), actions: (N, T, act_dim) ->
  obs windows (W, obs_horizon, obs_dim), action windows (W, pred_horizon, act_dim)
  """
  obs_w, act_w = [], []
  for o, a in zip(obs, actions):
    ix = window_indices(len(o), obs_horizon, pred_horizon, action_horizon)
    obs_w.append(o[ix[:, :obs_horizon]])
    act_w.append(a[ix])
  return np.concatenate(obs_w), np.concatenate(act_w)
