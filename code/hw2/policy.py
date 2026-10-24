"""Diffusion Policy wrapper: normalization, loss, action sampling, EMA, controller."""

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
  """p(A_t:t+Tp | O_t-To+1:t) via DDPM on min-max normalized actions.

  Observation/action normalizer statistics are registered buffers, so a
  checkpoint's state_dict is self-contained.
  """

  def __init__(self, obs_dim, act_dim, obs_horizon=2, pred_horizon=16, action_horizon=8,
               down_dims=(128, 256, 512), diffusion_step_embed_dim=128, kernel_size=5,
               n_groups=8, num_train_timesteps=100, num_inference_steps=100):
    super().__init__()
    self.kwargs = dict(
        obs_dim=obs_dim, act_dim=act_dim, obs_horizon=obs_horizon, pred_horizon=pred_horizon,
        action_horizon=action_horizon, down_dims=list(down_dims),
        diffusion_step_embed_dim=diffusion_step_embed_dim, kernel_size=kernel_size,
        n_groups=n_groups, num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps)
    self.obs_dim, self.act_dim = obs_dim, act_dim
    self.obs_horizon, self.pred_horizon, self.action_horizon = obs_horizon, pred_horizon, action_horizon
    self.num_inference_steps = num_inference_steps
    self.model = ConditionalUnet1D(
        input_dim=act_dim, global_cond_dim=obs_dim * obs_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
        kernel_size=kernel_size, n_groups=n_groups)
    self.noise_scheduler = NoiseScheduler(num_train_timesteps)
    for name, dim in [("obs", obs_dim), ("act", act_dim)]:
      self.register_buffer(f"{name}_scale", torch.ones(dim))
      self.register_buffer(f"{name}_offset", torch.zeros(dim))

  def set_normalizers(self, obs_norm: MinMaxNormalizer, act_norm: MinMaxNormalizer):
    for name, norm in [("obs", obs_norm), ("act", act_norm)]:
      getattr(self, f"{name}_scale").copy_(torch.from_numpy(norm.scale))
      getattr(self, f"{name}_offset").copy_(torch.from_numpy(norm.offset))

  def normalize_obs(self, obs):
    return obs * self.obs_scale + self.obs_offset

  def unnormalize_act(self, nact):
    return (nact - self.act_offset) / self.act_scale

  def compute_loss(self, nobs, nact, generator=None):
    """Epsilon-prediction MSE on already-normalized (B, To, D) obs and (B, Tp, A) actions."""
    B = nact.shape[0]
    noise = torch.randn(nact.shape, device=nact.device, generator=generator)
    t = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,), device=nact.device,
                      generator=generator)
    noisy = self.noise_scheduler.add_noise(nact, noise, t)
    pred = self.model(noisy, t, nobs.flatten(1))
    return F.mse_loss(pred, noise)

  @torch.no_grad()
  def predict_action(self, obs_hist, generator=None):
    """obs_hist: raw (B, To, obs_dim) tensor -> raw (B, Tp, act_dim) action sequence."""
    cond = self.normalize_obs(obs_hist).flatten(1)
    nact = self.noise_scheduler.sample(
        lambda x, t: self.model(x, t, cond),
        (obs_hist.shape[0], self.pred_horizon, self.act_dim), obs_hist.device,
        num_inference_steps=self.num_inference_steps, generator=generator)
    return self.unnormalize_act(nact)


class EMA:
  """Exponential moving average of weights with the warmup schedule used by
  Diffusion Policy (diffusers EMAModel, power=0.75): decay = 1 - (1 + step)^-power."""

  def __init__(self, model, power=0.75, max_decay=0.9999):
    self.model = copy.deepcopy(model).eval()
    for p in self.model.parameters():
      p.requires_grad_(False)
    self.power, self.max_decay, self.step = power, max_decay, 0

  @torch.no_grad()
  def update(self, model):
    decay = min(self.max_decay, 1.0 - (1.0 + self.step) ** -self.power)
    self.step += 1
    for ema_p, p in zip(self.model.parameters(), model.parameters()):
      ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(self.model.buffers(), model.buffers()):
      ema_b.copy_(b)


def atomic_torch_save(obj, path):
  """Write to a temporary file, then rename: an interrupted save never corrupts `path`."""
  path = str(path)
  tmp = path + ".tmp"
  torch.save(obj, tmp)
  os.replace(tmp, path)


def save_checkpoint(path, policy, extra=None):
  atomic_torch_save({"kwargs": policy.kwargs, "state_dict": policy.state_dict(), **(extra or {})}, path)


def load_checkpoint(path, device):
  ckpt = torch.load(path, map_location=device, weights_only=False)
  policy = DiffusionPolicy(**ckpt["kwargs"]).to(device)
  policy.load_state_dict(ckpt["state_dict"])
  policy.eval()
  return policy, ckpt


class DiffusionPolicyController:
  """Receding-horizon execution in a batched env.

  Keeps the last `obs_horizon` observations (the first one is repeated at
  t = 0, matching the edge padding used in training). Every `action_horizon`
  steps it samples a new (Tp, A) action sequence and executes actions
  [To - 1, To - 1 + Ta), i.e. those aligned with the current time step.
  """

  def __init__(self, policy, device, seed=0, action_horizon=None):
    self.policy = policy
    self.device = device
    self.To = policy.obs_horizon
    self.Ta = action_horizon or policy.action_horizon
    self.generator = torch.Generator(device=device)
    self.generator.manual_seed(seed)
    self.num_inferences = 0

  def reset(self, snap):
    obs = snap["obs"].astype(np.float32)
    self.history = deque([obs] * self.To, maxlen=self.To)
    self.queue = []

  def act(self, snap, t):
    if t > 0:
      self.history.append(snap["obs"].astype(np.float32))
    if not self.queue:
      obs_hist = torch.from_numpy(np.stack(self.history, axis=1)).to(self.device)
      actions = self.policy.predict_action(obs_hist, generator=self.generator).cpu().numpy()
      start = self.To - 1
      self.queue = [actions[:, k] for k in range(start, start + self.Ta)]
      self.num_inferences += 1
    return np.clip(self.queue.pop(0), -1.0, 1.0)
