"""Config loading, seeding, JSON I/O, statistics and provenance helpers."""

import copy
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def resolve_path(path):
  """Resolve a path relative to the repository root (absolute paths pass through)."""
  path = Path(path)
  return path if path.is_absolute() else REPO_ROOT / path


def deep_merge(base, override):
  out = copy.deepcopy(base)
  for k, v in override.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = deep_merge(out[k], v)
    else:
      out[k] = copy.deepcopy(v)
  return out


def set_dotted(cfg, key, value):
  node = cfg
  parts = key.split(".")
  for p in parts[:-1]:
    node = node.setdefault(p, {})
  node[parts[-1]] = value


def load_config(path, overrides=()):
  """Load a YAML config. A top-level `base:` key inherits from another config.

  `overrides` are "dotted.key=value" strings; values are parsed as YAML, so
  `train.num_steps=2000`, `seed=1` and `policy.down_dims=[64,128]` all work.
  """
  with open(resolve_path(path)) as f:
    cfg = yaml.safe_load(f) or {}
  if "base" in cfg:
    cfg = deep_merge(load_config(cfg.pop("base")), cfg)
  for ov in overrides:
    if "=" not in ov:
      raise ValueError(f"Override must look like key=value, got {ov!r}")
    key, value = ov.split("=", 1)
    set_dotted(cfg, key.strip(), yaml.safe_load(value))
  return cfg


def add_config_args(parser, default_config):
  parser.add_argument("--config", default=default_config, help="YAML config path")
  parser.add_argument(
      "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
      help="Config overrides, e.g. --set seed=1 train.num_steps=5000")


def save_yaml(obj, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    yaml.safe_dump(obj, f, sort_keys=False)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _json_default(o):
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return float(o)
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, Path):
    return str(o)
  raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
  with open(path) as f:
    return json.load(f)


# ---------------------------------------------------------------------------
# Seeding and statistics
# ---------------------------------------------------------------------------

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  try:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  except ImportError:
    pass


def wilson_interval(k, n, z=1.96):
  """95% Wilson score interval for a binomial proportion k/n."""
  if n == 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1 + z * z / n
  center = (p + z * z / (2 * n)) / denom
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
  return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

_PACKAGES = ["numpy", "torch", "jax", "jaxlib", "jax-cuda12-plugin", "mujoco",
             "mujoco-mjx", "playground", "brax", "flax", "imageio", "pyyaml"]


def git_commit():
  try:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=10)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "code", "configs"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    commit = out.stdout.strip() or None
    if commit and dirty.stdout.strip():
      commit += "-dirty"
    return commit
  except (OSError, subprocess.SubprocessError):
    return None


def env_info():
  """Hardware / library versions recorded alongside every result."""
  info = {
      "python": sys.version.split()[0],
      "platform": platform.platform(),
      "hostname": platform.node(),
      "git_commit": git_commit(),
      "packages": {},
      "env_vars": {k: os.environ.get(k) for k in ["MUJOCO_GL", "XLA_PYTHON_CLIENT_PREALLOCATE"]},
  }
  for pkg in _PACKAGES:
    try:
      info["packages"][pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
      pass
  try:
    import torch
    if torch.cuda.is_available():
      info["gpu"] = torch.cuda.get_device_name(0)
      info["torch_cuda"] = torch.version.cuda
  except ImportError:
    pass
  return info
