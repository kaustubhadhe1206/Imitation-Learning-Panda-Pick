"""Loading demonstrations and cutting them into training windows for Diffusion Policy.

The windowing is the same as in Chi et al.'s state-based Diffusion Policy
notebook: each training sample is a sequence of `pred_horizon` actions starting
at time t, together with the `obs_horizon` observations starting at t.
At the start of an episode the first observation is repeated, and at the end
the last action is repeated, so every step can be used. Here that padding is
done by clipping the time indices to [0, episode_length - 1].
"""

import numpy as np


def load_demonstrations(path, num_demos=None, only_successful=True):
  """Load demos from an .npz file. With num_demos=N we take the first N successful demos by seed.

  Because we always take the first N in seed order, the 25-demo set is part of
  the 50-demo set, which is part of the 100-demo set.
  """
  with np.load(path) as saved:
    if only_successful:
      episode_ids = np.flatnonzero(saved["success"])
    else:
      episode_ids = np.arange(len(saved["seeds"]))
    episode_ids = episode_ids[np.argsort(saved["seeds"][episode_ids], kind="stable")]
    if num_demos is not None:
      if len(episode_ids) < num_demos:
        raise ValueError(f"Asked for {num_demos} demos but {path} only has {len(episode_ids)}")
      episode_ids = episode_ids[:num_demos]
    return {
        "obs": saved["obs"][episode_ids].astype(np.float32),
        "actions": saved["actions"][episode_ids].astype(np.float32),
        "seeds": saved["seeds"][episode_ids],
    }


class MinMaxNormalizer:
  """Scales every dimension so that its min..max range becomes -1..1."""

  def __init__(self, data_min, data_max, eps=1e-4):
    self.min = np.asarray(data_min, dtype=np.float32)
    self.max = np.asarray(data_max, dtype=np.float32)
    data_range = self.max - self.min
    # Dimensions that (almost) never change would divide by ~0, so map them to 0 instead.
    changes = data_range > eps
    self.scale = np.where(changes, 2.0 / np.maximum(data_range, eps), 1.0).astype(np.float32)
    self.offset = np.where(changes, -1.0 - self.min * self.scale, -self.min).astype(np.float32)

  @classmethod
  def fit(cls, data):
    flat = data.reshape(-1, data.shape[-1])
    return cls(flat.min(axis=0), flat.max(axis=0))

  def normalize(self, data):
    return data * self.scale + self.offset

  def unnormalize(self, normalized):
    return (normalized - self.offset) / self.scale


def get_window_time_indices(episode_length, obs_horizon, pred_horizon, action_horizon):
  """Time indices of every training window in one episode, shape (num_windows, pred_horizon)."""
  window_starts = np.arange(-(obs_horizon - 1), episode_length - pred_horizon + action_horizon)
  time_indices = window_starts[:, None] + np.arange(pred_horizon)[None, :]
  return np.clip(time_indices, 0, episode_length - 1)


def make_training_windows(obs, actions, obs_horizon, pred_horizon, action_horizon):
  """Cut all episodes into windows.

  obs: (num_episodes, T, obs_dim), actions: (num_episodes, T, action_dim)
  returns obs windows (num_windows, obs_horizon, obs_dim) and
  action windows (num_windows, pred_horizon, action_dim)
  """
  obs_windows, action_windows = [], []
  for episode_obs, episode_actions in zip(obs, actions):
    time_indices = get_window_time_indices(len(episode_obs), obs_horizon, pred_horizon, action_horizon)
    obs_windows.append(episode_obs[time_indices[:, :obs_horizon]])
    action_windows.append(episode_actions[time_indices])
  return np.concatenate(obs_windows), np.concatenate(action_windows)
