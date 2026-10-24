"""Small helpers: configs, saving/loading files, seeds, statistics, and system info."""

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

# Top folder of the repository (this file is code/hw2/utils.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Config files
# ----------------------------------------------------------------------------

def repo_path(path):
  """Turn a path relative to the repo root into a full path (full paths are kept as they are)."""
  path = Path(path)
  return path if path.is_absolute() else REPO_ROOT / path


def merge_configs(base_config, override_config):
  """Return a copy of base_config with the values from override_config put on top (nested dicts too)."""
  merged = copy.deepcopy(base_config)
  for key, value in override_config.items():
    if isinstance(value, dict) and isinstance(merged.get(key), dict):
      merged[key] = merge_configs(merged[key], value)
    else:
      merged[key] = copy.deepcopy(value)
  return merged


def set_nested_value(config, dotted_key, value):
  """set_nested_value(cfg, "train.lr", 0.001) does cfg["train"]["lr"] = 0.001."""
  node = config
  key_parts = dotted_key.split(".")
  for part in key_parts[:-1]:
    node = node.setdefault(part, {})
  node[key_parts[-1]] = value


def load_config(path, overrides=()):
  """Load a YAML config file.

  If the file has a `base:` entry, that config is loaded first and this file
  is applied on top of it. `overrides` are strings like "train.num_steps=2000"
  (the value is read as YAML, so numbers and lists like "[64,128]" work).
  """
  with open(repo_path(path)) as f:
    config = yaml.safe_load(f) or {}
  if "base" in config:
    config = merge_configs(load_config(config.pop("base")), config)
  for override in overrides:
    if "=" not in override:
      raise ValueError(f"Override must look like key=value, got {override!r}")
    key, value = override.split("=", 1)
    set_nested_value(config, key.strip(), yaml.safe_load(value))
  return config


def add_config_arguments(parser, default_config_path):
  """Add the --config and --set command line options that every script uses."""
  parser.add_argument("--config", default=default_config_path, help="YAML config file")
  parser.add_argument(
      "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
      help="change config values, e.g. --set seed=1 train.num_steps=5000")


def save_yaml(data, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)


# ----------------------------------------------------------------------------
# JSON files
# ----------------------------------------------------------------------------

def _to_json_type(value):
  """json.dump can't handle numpy numbers/arrays or Paths, so convert them."""
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, Path):
    return str(value)
  raise TypeError(f"Can't save {type(value)} to JSON")


def save_json(data, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=2, default=_to_json_type)


def load_json(path):
  with open(path) as f:
    return json.load(f)


# ----------------------------------------------------------------------------
# Random seeds and statistics
# ----------------------------------------------------------------------------

def set_random_seed(seed):
  """Seed python, numpy and torch (if installed) so runs are repeatable."""
  random.seed(seed)
  np.random.seed(seed)
  try:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  except ImportError:
    pass


def wilson_confidence_interval(num_success, num_trials, z=1.96):
  """95% confidence interval for a success rate (Wilson score interval).

  Works better than the usual p +- 1.96 * std when the rate is close to 0 or 1.
  """
  if num_trials == 0:
    return (float("nan"), float("nan"))
  rate = num_success / num_trials
  denominator = 1 + z * z / num_trials
  center = (rate + z * z / (2 * num_trials)) / denominator
  half_width = z * math.sqrt(rate * (1 - rate) / num_trials + z * z / (4 * num_trials ** 2)) / denominator
  return (max(0.0, center - half_width), min(1.0, center + half_width))


# ----------------------------------------------------------------------------
# System info saved next to every result (for the reproducibility section)
# ----------------------------------------------------------------------------

PACKAGES_TO_RECORD = ["numpy", "torch", "jax", "jaxlib", "jax-cuda12-plugin", "mujoco",
                      "mujoco-mjx", "playground", "brax", "flax", "imageio", "pyyaml"]


def get_git_commit():
  """Short hash of the current commit, with "-dirty" added if code/ or configs/ have unsaved edits."""
  try:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=10)
    changes = subprocess.run(["git", "status", "--porcelain", "--", "code", "configs"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    commit = head.stdout.strip() or None
    if commit and changes.stdout.strip():
      commit += "-dirty"
    return commit
  except (OSError, subprocess.SubprocessError):
    return None


def get_system_info():
  """Python version, OS, GPU, package versions and git commit."""
  info = {
      "python": sys.version.split()[0],
      "platform": platform.platform(),
      "hostname": platform.node(),
      "git_commit": get_git_commit(),
      "packages": {},
      "env_vars": {name: os.environ.get(name) for name in ["MUJOCO_GL", "XLA_PYTHON_CLIENT_PREALLOCATE"]},
  }
  for package in PACKAGES_TO_RECORD:
    try:
      info["packages"][package] = importlib.metadata.version(package)
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
