"""Train a state-based Diffusion Policy on scripted-expert demonstrations.

Writes to --run-dir (default runs/<num_demos>_s<seed>):
  checkpoint.pt         EMA weights + normalizer + model kwargs
  config.yaml           fully resolved config
  loss.csv              step, loss (mean over log interval), lr, elapsed seconds
  train_summary.json    dataset seeds used, window count, timing, versions
  loss_curve.png

Usage:
  python code/train.py --set data.num_demos=25 seed=1
  python code/train.py --set train.num_steps=300 --run-dir runs/debug      # quick check
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import hw2  # noqa: F401
from hw2 import plotting, utils
from hw2.dataset import MinMaxNormalizer, load_demos, make_windows
from hw2.policy import EMA, DiffusionPolicy, save_checkpoint


def lr_lambda(warmup, total):
  """Linear warmup then cosine decay to 0 (diffusers 'cosine' schedule)."""
  def f(step):
    if step < warmup:
      return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
  return f


def plot_loss(csv_rows, path, title):
  plotting.setup()
  plt = plotting.plt
  steps = np.array([r["step"] for r in csv_rows])
  loss = np.array([r["loss"] for r in csv_rows])
  fig, ax = plt.subplots(figsize=(6, 3.6))
  ax.plot(steps, loss, color=plotting.SERIES[0], lw=1.5)
  ax.set_yscale("log")
  ax.set(xlabel="gradient step", ylabel="noise-prediction MSE", title=title)
  fig.tight_layout()
  plotting.save(fig, path)


def train(cfg, run_dir):
  run_dir = Path(run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  utils.set_seed(cfg["seed"])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pc, tc = cfg["policy"], cfg["train"]

  demos = load_demos(utils.resolve_path(cfg["data"]["path"]), cfg["data"]["num_demos"])
  obs_norm = MinMaxNormalizer.fit(demos["obs"])
  act_norm = MinMaxNormalizer.fit(demos["actions"])
  obs_w, act_w = make_windows(demos["obs"], demos["actions"], pc["obs_horizon"],
                              pc["pred_horizon"], pc["action_horizon"])
  nobs = torch.from_numpy(obs_norm.normalize(obs_w)).to(device)
  nact = torch.from_numpy(act_norm.normalize(act_w)).to(device)
  W = len(nobs)
  print(f"{len(demos['seeds'])} demos -> {W} windows | obs {tuple(obs_w.shape[1:])} "
        f"act {tuple(act_w.shape[1:])} | device {device}", flush=True)

  policy = DiffusionPolicy(obs_dim=obs_w.shape[-1], act_dim=act_w.shape[-1], **pc).to(device)
  policy.set_normalizers(obs_norm, act_norm)
  ema = EMA(policy, power=tc["ema_power"])
  n_params = sum(p.numel() for p in policy.parameters())
  print(f"policy parameters: {n_params / 1e6:.2f} M", flush=True)

  opt = torch.optim.AdamW(policy.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc["warmup_steps"], tc["num_steps"]))
  gen = torch.Generator(device=device)
  gen.manual_seed(cfg["seed"])

  rows, running, t0 = [], [], time.time()
  policy.train()
  for step in range(1, tc["num_steps"] + 1):
    idx = torch.randint(0, W, (tc["batch_size"],), device=device, generator=gen)
    loss = policy.compute_loss(nobs[idx], nact[idx], generator=gen)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    ema.update(policy)
    running.append(loss.detach())  # no per-step GPU sync
    if step % tc["log_every"] == 0 or step == tc["num_steps"]:
      row = {"step": step, "loss": torch.stack(running).mean().item(), "lr": sched.get_last_lr()[0],
             "elapsed_s": time.time() - t0}
      rows.append(row)
      running = []
      if step % (tc["log_every"] * 10) == 0 or step == tc["num_steps"]:
        rate = step / row["elapsed_s"]
        print(f"step {step:6d} loss {row['loss']:.5f} lr {row['lr']:.2e} "
              f"{rate:.1f} it/s eta {(tc['num_steps'] - step) / rate / 60:.1f} min", flush=True)

  train_time = time.time() - t0
  save_checkpoint(run_dir / "checkpoint.pt", ema.model, extra={"config": cfg, "step": tc["num_steps"]})
  utils.save_yaml(cfg, run_dir / "config.yaml")
  with open(run_dir / "loss.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = {
      "num_demos": len(demos["seeds"]),
      "demo_seeds": demos["seeds"].tolist(),
      "num_windows": W,
      "obs_dim": int(obs_w.shape[-1]),
      "act_dim": int(act_w.shape[-1]),
      "num_params": n_params,
      "train_time_s": train_time,
      "final_loss": rows[-1]["loss"],
      "seed": cfg["seed"],
      "device": str(device),
      "env_info": utils.env_info(),
  }
  utils.save_json(summary, run_dir / "train_summary.json")
  plot_loss(rows, run_dir / "loss_curve.png",
            f"Training loss ({summary['num_demos']} demos, seed {cfg['seed']})")
  print(json.dumps({k: summary[k] for k in ["num_demos", "num_windows", "num_params", "train_time_s",
                                            "final_loss"]}, indent=2))
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  utils.add_config_args(parser, "configs/dp_base.yaml")
  parser.add_argument("--run-dir", default=None)
  args = parser.parse_args()
  cfg = utils.load_config(args.config, args.overrides)
  run_dir = args.run_dir or f"runs/n{cfg['data']['num_demos']}_s{cfg['seed']}"
  train(cfg, utils.resolve_path(run_dir))


if __name__ == "__main__":
  main()
