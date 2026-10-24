"""Noise-prediction network (1-D conditional U-Net) and the diffusion noise scheduler.

The network is adapted from the official Diffusion Policy code (Chi et al.,
"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023,
https://github.com/real-stanford/diffusion_policy, MIT License). The class
names and layer structure are kept the same as in their state-based notebook
so it's easy to compare. What I changed:
  - the channel widths are configurable (we use smaller ones than the notebook),
  - local variable names were renamed to be easier to read,
  - instead of using the `diffusers` library for the noise scheduler, I wrote
    the DDPM/DDIM update rules myself below (same formulas as diffusers'
    DDPMScheduler/DDIMScheduler with the squaredcos_cap_v2 schedule, epsilon
    prediction, and clipping the predicted clean sample to [-1, 1]). This
    removes a dependency and lets me unit-test the sampler.
"""

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Network (adapted from Chi et al.)
# ----------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
  """Turns the diffusion step number into a vector of sines and cosines."""

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, timesteps):
    half_dim = self.dim // 2
    frequency_step = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -frequency_step)
    angles = timesteps[:, None].float() * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Downsample1d(nn.Module):
  """Halves the sequence length."""

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv1d(channels, channels, 3, 2, 1)

  def forward(self, x):
    return self.conv(x)


class Upsample1d(nn.Module):
  """Doubles the sequence length."""

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)

  def forward(self, x):
    return self.conv(x)


class Conv1dBlock(nn.Module):
  """Conv1d -> GroupNorm -> Mish."""

  def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.GroupNorm(n_groups, out_channels),
        nn.Mish(),
    )

  def forward(self, x):
    return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
  """Two conv blocks; in between, the condition scales and shifts every channel (FiLM)."""

  def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
    super().__init__()
    self.blocks = nn.ModuleList([
        Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
        Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
    ])
    self.out_channels = out_channels
    # The condition gives one scale and one shift per output channel.
    self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
    if in_channels != out_channels:
      self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
    else:
      self.residual_conv = nn.Identity()

  def forward(self, x, cond):
    out = self.blocks[0](x)
    film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
    scale, shift = film[:, 0], film[:, 1]
    out = scale * out + shift
    out = self.blocks[1](out)
    return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
  """Predicts the noise in a noisy action sequence, given the diffusion step and the observations."""

  def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=128,
               down_dims=(128, 256, 512), kernel_size=5, n_groups=8):
    super().__init__()
    channel_sizes = [input_dim] + list(down_dims)
    first_width = down_dims[0]
    step_dim = diffusion_step_embed_dim
    self.diffusion_step_encoder = nn.Sequential(
        SinusoidalPosEmb(step_dim), nn.Linear(step_dim, step_dim * 4), nn.Mish(),
        nn.Linear(step_dim * 4, step_dim))
    cond_dim = step_dim + global_cond_dim
    in_out_pairs = list(zip(channel_sizes[:-1], channel_sizes[1:]))
    middle_width = channel_sizes[-1]

    def residual_block(in_channels, out_channels):
      return ConditionalResidualBlock1D(in_channels, out_channels, cond_dim,
                                        kernel_size=kernel_size, n_groups=n_groups)

    self.mid_modules = nn.ModuleList([residual_block(middle_width, middle_width),
                                      residual_block(middle_width, middle_width)])
    self.down_modules = nn.ModuleList()
    for level, (in_channels, out_channels) in enumerate(in_out_pairs):
      is_last_level = level >= len(in_out_pairs) - 1
      self.down_modules.append(nn.ModuleList([
          residual_block(in_channels, out_channels),
          residual_block(out_channels, out_channels),
          Downsample1d(out_channels) if not is_last_level else nn.Identity()]))
    self.up_modules = nn.ModuleList()
    for level, (in_channels, out_channels) in enumerate(reversed(in_out_pairs[1:])):
      is_last_level = level >= len(in_out_pairs) - 1
      self.up_modules.append(nn.ModuleList([
          residual_block(out_channels * 2, in_channels),
          residual_block(in_channels, in_channels),
          Upsample1d(in_channels) if not is_last_level else nn.Identity()]))
    self.final_conv = nn.Sequential(
        Conv1dBlock(first_width, first_width, kernel_size=kernel_size, n_groups=n_groups),
        nn.Conv1d(first_width, input_dim, 1))

  def forward(self, noisy_actions, timestep, global_cond):
    """noisy_actions: (B, T, action_dim), timestep: (B,) ints, global_cond: (B, cond_dim)."""
    x = noisy_actions.moveaxis(-1, -2)     # conv layers want (B, channels, T)
    if not torch.is_tensor(timestep):
      timestep = torch.tensor([timestep], device=noisy_actions.device)
    timestep = timestep.expand(noisy_actions.shape[0])
    cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)
    skip_connections = []
    for block1, block2, downsample in self.down_modules:
      x = block2(block1(x, cond), cond)
      skip_connections.append(x)
      x = downsample(x)
    for block in self.mid_modules:
      x = block(x, cond)
    for block1, block2, upsample in self.up_modules:
      x = torch.cat((x, skip_connections.pop()), dim=1)
      x = block2(block1(x, cond), cond)
      x = upsample(x)
    return self.final_conv(x).moveaxis(-1, -2)


# ----------------------------------------------------------------------------
# Noise scheduler
# ----------------------------------------------------------------------------

def make_cosine_beta_schedule(num_steps, max_beta=0.999):
  """Cosine noise schedule (Nichol & Dhariwal 2021), "squaredcos_cap_v2" in diffusers."""
  def alpha_bar(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
  betas = [min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
           for i in range(num_steps)]
  return torch.tensor(betas, dtype=torch.float64)


class NoiseScheduler:
  """Adds noise for training, and removes it step by step for sampling.

  alpha_bars[t] is how much of the clean signal is left at step t
  (noisy = sqrt(alpha_bar) * clean + sqrt(1 - alpha_bar) * noise).
  """

  def __init__(self, num_train_timesteps=100, clip_sample=True):
    self.num_train_timesteps = num_train_timesteps
    self.clip_sample = clip_sample
    betas = make_cosine_beta_schedule(num_train_timesteps)
    self.alpha_bars = torch.cumprod(1.0 - betas, dim=0)   # float64, on CPU

  def add_noise(self, clean, noise, timesteps):
    alpha_bar = self.alpha_bars.to(device=clean.device, dtype=torch.float32)[timesteps]
    alpha_bar = alpha_bar.reshape(-1, *([1] * (clean.dim() - 1)))
    return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

  def _clip(self, predicted_clean):
    return predicted_clean.clamp(-1.0, 1.0) if self.clip_sample else predicted_clean

  def ddpm_denoise_step(self, predicted_noise, t, noisy, generator=None):
    """One DDPM step from step t to step t-1 (adds a bit of fresh noise, except at t = 0)."""
    alpha_bar_t = float(self.alpha_bars[t])
    alpha_bar_prev = float(self.alpha_bars[t - 1]) if t > 0 else 1.0
    alpha_t = alpha_bar_t / alpha_bar_prev
    beta_t = 1.0 - alpha_t
    predicted_clean = self._clip((noisy - math.sqrt(1 - alpha_bar_t) * predicted_noise) / math.sqrt(alpha_bar_t))
    clean_weight = math.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
    noisy_weight = math.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    mean = clean_weight * predicted_clean + noisy_weight * noisy
    if t == 0:
      return mean
    variance = max((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t, 1e-20)
    fresh_noise = torch.randn(noisy.shape, device=noisy.device, dtype=noisy.dtype, generator=generator)
    return mean + math.sqrt(variance) * fresh_noise

  def ddim_denoise_step(self, predicted_noise, t, t_prev, noisy):
    """One deterministic DDIM step from step t to step t_prev (can skip steps)."""
    alpha_bar_t = float(self.alpha_bars[t])
    alpha_bar_prev = float(self.alpha_bars[t_prev]) if t_prev >= 0 else 1.0
    predicted_clean = self._clip((noisy - math.sqrt(1 - alpha_bar_t) * predicted_noise) / math.sqrt(alpha_bar_t))
    predicted_noise = (noisy - math.sqrt(alpha_bar_t) * predicted_clean) / math.sqrt(1 - alpha_bar_t)
    return math.sqrt(alpha_bar_prev) * predicted_clean + math.sqrt(1 - alpha_bar_prev) * predicted_noise

  @torch.no_grad()
  def generate(self, predict_noise, shape, device, num_inference_steps=None, generator=None):
    """Start from pure noise and denoise it. predict_noise(noisy, timesteps) is the network.

    With num_inference_steps equal to the number of training steps we use the
    full DDPM chain; with fewer we use DDIM, which skips steps (faster).
    """
    num_train_steps = self.num_train_timesteps
    num_steps = num_inference_steps or num_train_steps
    sample = torch.randn(shape, device=device, generator=generator)
    if num_steps >= num_train_steps:
      for t in reversed(range(num_train_steps)):
        timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
        sample = self.ddpm_denoise_step(predict_noise(sample, timesteps), t, sample, generator)
    else:
      stride = num_train_steps // num_steps
      for t in list(range(0, num_train_steps, stride))[:num_steps][::-1]:
        timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
        sample = self.ddim_denoise_step(predict_noise(sample, timesteps), t, t - stride, sample)
    return sample
