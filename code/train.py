"""Train the state-based Diffusion Policy on the expert demonstrations (Part 2).

Files written to --run-dir (default runs/n<num_demos>_s<seed>):
  checkpoint.pt         averaged (EMA) weights + normalization + model settings
  config.yaml           the full config that was used
  loss.csv              step, loss (averaged over the logging interval), lr, seconds
  train_summary.json    which demo seeds were used, number of windows, timing, versions
  loss_curve.png
  train_state.pt        (only while training) everything needed to continue training,
                        saved every train.save_every steps. Running the same command
                        again continues from it; it's deleted when training finishes.

Usage:
  python code/train.py --set data.num_demos=25 seed=1
  python code/train.py --set train.num_steps=300 --run-dir runs/debug      # quick test
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.dataset import MinMaxNormalizer, load_demonstrations, make_training_windows
from hw2.policy import DiffusionPolicy, ExponentialMovingAverage, safe_torch_save, save_checkpoint


def warmup_then_cosine(warmup_steps, total_steps):
  """Learning-rate multiplier: goes up linearly during warmup, then down to 0 along a cosine."""
  def lr_multiplier(step):
    if step < warmup_steps:
      return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
  return lr_multiplier


def plot_loss_curve(loss_rows, path, title):
  plotting.use_report_style()
  plt = plotting.plt
  steps = np.array([row["step"] for row in loss_rows])
  losses = np.array([row["loss"] for row in loss_rows])
  fig, ax = plt.subplots(figsize=(6, 3.6))
  ax.plot(steps, losses, color=plotting.COLORS[0], lw=1.5)
  ax.set_yscale("log")
  ax.set(xlabel="gradient step", ylabel="noise-prediction MSE", title=title)
  fig.tight_layout()
  plotting.save_figure(fig, path)


def settings_that_must_match(config):
  """If any of these changed, an old train_state.pt can't be continued."""
  train_config = config["train"]
  return (config["seed"], config["data"], config["policy"],
          {key: train_config[key] for key in ["num_steps", "batch_size", "lr", "weight_decay",
                                              "warmup_steps", "ema_power"]})


def train(config, run_dir, stop_after=None):
  run_dir = Path(run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  utils.set_random_seed(config["seed"])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  policy_config, train_config = config["policy"], config["train"]

  # --- data ---
  demos = load_demonstrations(utils.repo_path(config["data"]["path"]), config["data"]["num_demos"])
  obs_normalizer = MinMaxNormalizer.fit(demos["obs"])
  action_normalizer = MinMaxNormalizer.fit(demos["actions"])
  obs_windows, action_windows = make_training_windows(
      demos["obs"], demos["actions"], policy_config["obs_horizon"],
      policy_config["pred_horizon"], policy_config["action_horizon"])
  normalized_obs = torch.from_numpy(obs_normalizer.normalize(obs_windows)).to(device)
  normalized_actions = torch.from_numpy(action_normalizer.normalize(action_windows)).to(device)
  num_windows = len(normalized_obs)
  print(f"{len(demos['seeds'])} demos -> {num_windows} windows | obs {tuple(obs_windows.shape[1:])} "
        f"actions {tuple(action_windows.shape[1:])} | device {device}", flush=True)

  # --- model, optimizer ---
  policy = DiffusionPolicy(obs_dim=obs_windows.shape[-1], action_dim=action_windows.shape[-1],
                           **policy_config).to(device)
  policy.set_normalizers(obs_normalizer, action_normalizer)
  ema = ExponentialMovingAverage(policy, power=train_config["ema_power"])
  num_parameters = sum(param.numel() for param in policy.parameters())
  print(f"policy parameters: {num_parameters / 1e6:.2f} M", flush=True)

  optimizer = torch.optim.AdamW(policy.parameters(), lr=train_config["lr"],
                                weight_decay=train_config["weight_decay"])
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer, warmup_then_cosine(train_config["warmup_steps"], train_config["num_steps"]))
  # All the randomness in training (which windows, which noise, which diffusion step) comes from this.
  rng = torch.Generator(device=device)
  rng.manual_seed(config["seed"])

  # --- continue from a saved training state if there is one (e.g. after a Colab disconnect) ---
  state_path = run_dir / "train_state.pt"
  start_step, loss_rows, seconds_before = 0, [], 0.0
  if state_path.exists():
    saved_state = torch.load(state_path, map_location=device, weights_only=False)
    if settings_that_must_match(saved_state["config"]) != settings_that_must_match(config):
      raise RuntimeError(f"{state_path} was saved with a different config; delete it to start over.")
    policy.load_state_dict(saved_state["policy"])
    ema.averaged_model.load_state_dict(saved_state["ema"])
    ema.num_updates = saved_state["ema_num_updates"]
    optimizer.load_state_dict(saved_state["optimizer"])
    lr_scheduler.load_state_dict(saved_state["lr_scheduler"])
    rng.set_state(saved_state["rng"].cpu())
    start_step = saved_state["step"]
    loss_rows = saved_state["loss_rows"]
    seconds_before = saved_state["elapsed_s"]
    print(f"continuing from step {start_step} ({state_path})", flush=True)

  def save_training_state(step):
    safe_torch_save({
        "step": step, "config": config, "loss_rows": loss_rows,
        "elapsed_s": seconds_before + time.time() - start_time,
        "policy": policy.state_dict(), "ema": ema.averaged_model.state_dict(),
        "ema_num_updates": ema.num_updates, "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(), "rng": rng.get_state(),
    }, state_path)

  # --- training loop ---
  num_steps = train_config["num_steps"]
  save_every = train_config.get("save_every", 0)
  recent_losses = []
  start_time = time.time()
  policy.train()
  for step in range(start_step + 1, num_steps + 1):
    batch_ids = torch.randint(0, num_windows, (train_config["batch_size"],), device=device, generator=rng)
    loss = policy.compute_loss(normalized_obs[batch_ids], normalized_actions[batch_ids], generator=rng)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    ema.update(policy)
    recent_losses.append(loss.detach())   # .item() every step would slow down the GPU

    if step % train_config["log_every"] == 0 or step == num_steps:
      row = {"step": step, "loss": torch.stack(recent_losses).mean().item(),
             "lr": lr_scheduler.get_last_lr()[0], "elapsed_s": seconds_before + time.time() - start_time}
      loss_rows.append(row)
      recent_losses = []
      if step % (train_config["log_every"] * 10) == 0 or step == num_steps:
        steps_per_second = (step - start_step) / (time.time() - start_time)
        minutes_left = (num_steps - step) / steps_per_second / 60
        print(f"step {step:6d} loss {row['loss']:.5f} lr {row['lr']:.2e} "
              f"{steps_per_second:.1f} it/s eta {minutes_left:.1f} min", flush=True)

    stop_now = stop_after is not None and step - start_step >= stop_after and step < num_steps
    if (save_every and step % save_every == 0 and step < num_steps) or stop_now:
      save_training_state(step)
    if stop_now:
      print(f"stopped after {stop_after} steps at step {step}; state saved to {state_path}", flush=True)
      return None

  # --- save results ---
  train_seconds = seconds_before + time.time() - start_time
  save_checkpoint(run_dir / "checkpoint.pt", ema.averaged_model, extra={"config": config, "step": num_steps})
  state_path.unlink(missing_ok=True)   # not needed anymore, the final checkpoint is saved
  utils.save_yaml(config, run_dir / "config.yaml")
  with open(run_dir / "loss.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(loss_rows[0]))
    writer.writeheader()
    writer.writerows(loss_rows)
  summary = {
      "num_demos": len(demos["seeds"]),
      "demo_seeds": demos["seeds"].tolist(),
      "num_windows": num_windows,
      "obs_dim": int(obs_windows.shape[-1]),
      "action_dim": int(action_windows.shape[-1]),
      "num_params": num_parameters,
      "train_time_s": train_seconds,
      "final_loss": loss_rows[-1]["loss"],
      "seed": config["seed"],
      "device": str(device),
      "system_info": utils.get_system_info(),
  }
  utils.save_json(summary, run_dir / "train_summary.json")
  plot_loss_curve(loss_rows, run_dir / "loss_curve.png",
                  f"Training loss ({summary['num_demos']} demos, seed {config['seed']})")
  print(json.dumps({key: summary[key] for key in ["num_demos", "num_windows", "num_params",
                                                  "train_time_s", "final_loss"]}, indent=2))
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  utils.add_config_arguments(parser, "configs/dp_base.yaml")
  parser.add_argument("--run-dir", default=None)
  parser.add_argument("--stop-after", type=int, default=None,
                      help="save the training state and quit after this many steps (continue later)")
  args = parser.parse_args()
  config = utils.load_config(args.config, args.overrides)
  run_dir = args.run_dir or f"runs/n{config['data']['num_demos']}_s{config['seed']}"
  if train(config, utils.repo_path(run_dir), stop_after=args.stop_after) is None:
    sys.exit(3)   # stopped early on purpose, so there is no final checkpoint yet


if __name__ == "__main__":
  main()
