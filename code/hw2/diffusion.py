"""1-D temporal conditional U-Net and DDPM/DDIM noise scheduler.

The network is adapted from the official Diffusion Policy state-based notebook
(Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action
Diffusion", RSS 2023; https://github.com/real-stanford/diffusion_policy, MIT
License): ConditionalUnet1D with FiLM conditioning on the diffusion step
embedding concatenated with the flattened observation history. Changes: the
layer structure is unchanged; widths are configurable (we use a smaller
default than the notebook), and the noise scheduler below is a self-contained
re-implementation of the `diffusers` DDPMScheduler/DDIMScheduler update rules
(squaredcos_cap_v2 betas, epsilon prediction, x0 clipping to [-1, 1]) so the
project has no `diffusers` dependency and the sampler is unit-testable.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Network (adapted from Chi et al.)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    half = self.dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=x.device) * -emb)
    emb = x[:, None].float() * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

  def forward(self, x):
    return self.conv(x)


class Upsample1d(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

  def forward(self, x):
    return self.conv(x)


class Conv1dBlock(nn.Module):
  """Conv1d -> GroupNorm -> Mish."""

  def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.GroupNorm(n_groups, out_channels),
        nn.Mish(),
    )

  def forward(self, x):
    return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
  """Two conv blocks with FiLM (per-channel scale and bias) from the condition."""

  def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
    super().__init__()
    self.blocks = nn.ModuleList([
        Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
        Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
    ])
    self.out_channels = out_channels
    self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
    self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                          if in_channels != out_channels else nn.Identity())

  def forward(self, x, cond):
    out = self.blocks[0](x)
    embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
    out = embed[:, 0] * out + embed[:, 1]
    out = self.blocks[1](out)
    return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):

  def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=128,
               down_dims=(128, 256, 512), kernel_size=5, n_groups=8):
    super().__init__()
    all_dims = [input_dim] + list(down_dims)
    start_dim = down_dims[0]
    dsed = diffusion_step_embed_dim
    self.diffusion_step_encoder = nn.Sequential(
        SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
    cond_dim = dsed + global_cond_dim
    in_out = list(zip(all_dims[:-1], all_dims[1:]))
    mid_dim = all_dims[-1]

    def crb(i, o):
      return ConditionalResidualBlock1D(i, o, cond_dim, kernel_size=kernel_size, n_groups=n_groups)

    self.mid_modules = nn.ModuleList([crb(mid_dim, mid_dim), crb(mid_dim, mid_dim)])
    self.down_modules = nn.ModuleList()
    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= len(in_out) - 1
      self.down_modules.append(nn.ModuleList([
          crb(dim_in, dim_out), crb(dim_out, dim_out),
          Downsample1d(dim_out) if not is_last else nn.Identity()]))
    self.up_modules = nn.ModuleList()
    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= len(in_out) - 1
      self.up_modules.append(nn.ModuleList([
          crb(dim_out * 2, dim_in), crb(dim_in, dim_in),
          Upsample1d(dim_in) if not is_last else nn.Identity()]))
    self.final_conv = nn.Sequential(
        Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
        nn.Conv1d(start_dim, input_dim, 1))

  def forward(self, sample, timestep, global_cond):
    """sample: (B, T, input_dim); timestep: (B,) int; global_cond: (B, cond)."""
    x = sample.moveaxis(-1, -2)
    if not torch.is_tensor(timestep):
      timestep = torch.tensor([timestep], device=sample.device)
    timestep = timestep.expand(sample.shape[0])
    cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)
    skips = []
    for resnet, resnet2, downsample in self.down_modules:
      x = resnet2(resnet(x, cond), cond)
      skips.append(x)
      x = downsample(x)
    for mid in self.mid_modules:
      x = mid(x, cond)
    for resnet, resnet2, upsample in self.up_modules:
      x = torch.cat((x, skips.pop()), dim=1)
      x = resnet2(resnet(x, cond), cond)
      x = upsample(x)
    return self.final_conv(x).moveaxis(-1, -2)


# ---------------------------------------------------------------------------
# Noise scheduler
# ---------------------------------------------------------------------------

def cosine_betas(num_steps, max_beta=0.999):
  """squaredcos_cap_v2 schedule (Nichol & Dhariwal 2021), as in diffusers."""
  def alpha_bar(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
  return torch.tensor([min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
                       for i in range(num_steps)], dtype=torch.float64)


class NoiseScheduler:
  """DDPM training noise + DDPM (full chain) or DDIM (strided, eta=0) sampling."""

  def __init__(self, num_train_timesteps=100, clip_sample=True):
    self.num_train_timesteps = num_train_timesteps
    self.clip_sample = clip_sample
    betas = cosine_betas(num_train_timesteps)
    self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)   # float64 on CPU

  def _ab(self, t, device):
    return self.alphas_cumprod.to(device=device, dtype=torch.float32)[t]

  def add_noise(self, x0, noise, t):
    ab = self._ab(t, x0.device).reshape(-1, *([1] * (x0.dim() - 1)))
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

  def _clip(self, x0):
    return x0.clamp(-1.0, 1.0) if self.clip_sample else x0

  def ddpm_step(self, eps, t, x, generator=None):
    ab_t = float(self.alphas_cumprod[t])
    ab_prev = float(self.alphas_cumprod[t - 1]) if t > 0 else 1.0
    alpha_t = ab_t / ab_prev
    beta_t = 1.0 - alpha_t
    x0 = self._clip((x - math.sqrt(1 - ab_t) * eps) / math.sqrt(ab_t))
    mean = (math.sqrt(ab_prev) * beta_t / (1 - ab_t)) * x0 + (math.sqrt(alpha_t) * (1 - ab_prev) / (1 - ab_t)) * x
    if t == 0:
      return mean
    var = max((1 - ab_prev) / (1 - ab_t) * beta_t, 1e-20)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return mean + math.sqrt(var) * noise

  def ddim_step(self, eps, t, t_prev, x):
    ab_t = float(self.alphas_cumprod[t])
    ab_prev = float(self.alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0
    x0 = self._clip((x - math.sqrt(1 - ab_t) * eps) / math.sqrt(ab_t))
    eps = (x - math.sqrt(ab_t) * x0) / math.sqrt(1 - ab_t)
    return math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps

  @torch.no_grad()
  def sample(self, eps_fn, shape, device, num_inference_steps=None, generator=None):
    """Draw x0 ~ p(x0 | cond) given eps_fn(x_t, t_batch) -> predicted noise."""
    T = self.num_train_timesteps
    n = num_inference_steps or T
    x = torch.randn(shape, device=device, generator=generator)
    if n >= T:
      for t in reversed(range(T)):
        eps = eps_fn(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        x = self.ddpm_step(eps, t, x, generator)
    else:
      stride = T // n
      timesteps = list(range(0, T, stride))[:n][::-1]
      for t in timesteps:
        eps = eps_fn(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        x = self.ddim_step(eps, t, t - stride, x)
    return x
