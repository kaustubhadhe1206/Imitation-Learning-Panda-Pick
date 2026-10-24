"""Diffusion Policy: normalization, training loss, sampling actions, weight averaging,
checkpoints, and the controller that runs the policy in the env."""

import copy
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import MinMaxNormalizer
from .diffusion import ConditionalUnet1D, NoiseScheduler


class DiffusionPolicy(nn.Module):
  """Given the last `obs_horizon` observations, generate the next `pred_horizon` actions.

  The normalization numbers are stored as buffers inside the model, so a saved
  checkpoint has everything needed to run the policy.
  """

  def __init__(self, obs_dim, action_dim, obs_horizon=2, pred_horizon=16, action_horizon=8,
               down_dims=(128, 256, 512), diffusion_step_embed_dim=128, kernel_size=5,
               n_groups=8, num_train_timesteps=100, num_inference_steps=100):
    super().__init__()
    # Saved in the checkpoint so the same model can be rebuilt when loading.
    self.init_args = dict(
        obs_dim=obs_dim, action_dim=action_dim, obs_horizon=obs_horizon, pred_horizon=pred_horizon,
        action_horizon=action_horizon, down_dims=list(down_dims),
        diffusion_step_embed_dim=diffusion_step_embed_dim, kernel_size=kernel_size,
        n_groups=n_groups, num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps)
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.obs_horizon = obs_horizon
    self.pred_horizon = pred_horizon
    self.action_horizon = action_horizon
    self.num_inference_steps = num_inference_steps
    self.noise_predictor = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
        kernel_size=kernel_size, n_groups=n_groups)
    self.noise_scheduler = NoiseScheduler(num_train_timesteps)
    self.register_buffer("obs_norm_scale", torch.ones(obs_dim))
    self.register_buffer("obs_norm_offset", torch.zeros(obs_dim))
    self.register_buffer("action_norm_scale", torch.ones(action_dim))
    self.register_buffer("action_norm_offset", torch.zeros(action_dim))

  def set_normalizers(self, obs_normalizer: MinMaxNormalizer, action_normalizer: MinMaxNormalizer):
    self.obs_norm_scale.copy_(torch.from_numpy(obs_normalizer.scale))
    self.obs_norm_offset.copy_(torch.from_numpy(obs_normalizer.offset))
    self.action_norm_scale.copy_(torch.from_numpy(action_normalizer.scale))
    self.action_norm_offset.copy_(torch.from_numpy(action_normalizer.offset))

  def normalize_obs(self, obs):
    return obs * self.obs_norm_scale + self.obs_norm_offset

  def unnormalize_actions(self, normalized_actions):
    return (normalized_actions - self.action_norm_offset) / self.action_norm_scale

  def compute_loss(self, normalized_obs, normalized_actions, generator=None):
    """Training loss: add random noise to the actions and ask the network to predict that noise.

    normalized_obs: (B, obs_horizon, obs_dim), normalized_actions: (B, pred_horizon, action_dim)
    """
    batch_size = normalized_actions.shape[0]
    noise = torch.randn(normalized_actions.shape, device=normalized_actions.device, generator=generator)
    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,),
                              device=normalized_actions.device, generator=generator)
    noisy_actions = self.noise_scheduler.add_noise(normalized_actions, noise, timesteps)
    predicted_noise = self.noise_predictor(noisy_actions, timesteps, normalized_obs.flatten(1))
    return F.mse_loss(predicted_noise, noise)

  @torch.no_grad()
  def predict_action_sequence(self, obs_history, generator=None):
    """obs_history: raw (B, obs_horizon, obs_dim) -> raw actions (B, pred_horizon, action_dim)."""
    obs_condition = self.normalize_obs(obs_history).flatten(1)
    normalized_actions = self.noise_scheduler.generate(
        lambda noisy, timesteps: self.noise_predictor(noisy, timesteps, obs_condition),
        (obs_history.shape[0], self.pred_horizon, self.action_dim), obs_history.device,
        num_inference_steps=self.num_inference_steps, generator=generator)
    return self.unnormalize_actions(normalized_actions)


class ExponentialMovingAverage:
  """Keeps a slowly-updated average copy of the weights; we evaluate with this copy.

  Same warm-up as Diffusion Policy (diffusers EMAModel with power=0.75):
  decay = 1 - (1 + num_updates) ** -power, so early on it follows the model closely.
  """

  def __init__(self, model, power=0.75, max_decay=0.9999):
    self.averaged_model = copy.deepcopy(model).eval()
    for param in self.averaged_model.parameters():
      param.requires_grad_(False)
    self.power = power
    self.max_decay = max_decay
    self.num_updates = 0

  @torch.no_grad()
  def update(self, model):
    decay = min(self.max_decay, 1.0 - (1.0 + self.num_updates) ** -self.power)
    self.num_updates += 1
    for averaged_param, param in zip(self.averaged_model.parameters(), model.parameters()):
      averaged_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    for averaged_buffer, buffer in zip(self.averaged_model.buffers(), model.buffers()):
      averaged_buffer.copy_(buffer)


def safe_torch_save(data, path):
  """Save to a temporary file first and then rename it, so a crash mid-save never leaves a broken file."""
  path = str(path)
  temp_path = path + ".tmp"
  torch.save(data, temp_path)
  os.replace(temp_path, path)


def save_checkpoint(path, policy, extra=None):
  safe_torch_save({"init_args": policy.init_args, "state_dict": policy.state_dict(), **(extra or {})}, path)


def load_checkpoint(path, device):
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  policy = DiffusionPolicy(**checkpoint["init_args"]).to(device)
  policy.load_state_dict(checkpoint["state_dict"])
  policy.eval()
  return policy, checkpoint


class DiffusionPolicyController:
  """Runs the policy in the batched env, re-planning every `action_horizon` steps.

  It remembers the last `obs_horizon` observations (at the first step the
  first observation is repeated, same as the padding in training). When it runs
  out of planned actions it generates a new sequence of pred_horizon actions and
  executes the ones that line up with the current step:
  [obs_horizon - 1, obs_horizon - 1 + action_horizon).
  """

  def __init__(self, policy, device, seed=0, action_horizon=None):
    self.policy = policy
    self.device = device
    self.obs_horizon = policy.obs_horizon
    self.action_horizon = action_horizon or policy.action_horizon
    self.generator = torch.Generator(device=device)
    self.generator.manual_seed(seed)
    self.num_policy_calls = 0

  def reset(self, env_state):
    first_obs = env_state["obs"].astype(np.float32)
    self.obs_history = deque([first_obs] * self.obs_horizon, maxlen=self.obs_horizon)
    self.planned_actions = []

  def get_action(self, env_state, step):
    if step > 0:
      self.obs_history.append(env_state["obs"].astype(np.float32))
    if not self.planned_actions:
      obs_history = torch.from_numpy(np.stack(self.obs_history, axis=1)).to(self.device)
      action_sequence = self.policy.predict_action_sequence(obs_history, generator=self.generator)
      action_sequence = action_sequence.cpu().numpy()
      first = self.obs_horizon - 1
      self.planned_actions = [action_sequence[:, k] for k in range(first, first + self.action_horizon)]
      self.num_policy_calls += 1
    return np.clip(self.planned_actions.pop(0), -1.0, 1.0)
