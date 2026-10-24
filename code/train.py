"""Train a state-based Diffusion Policy on scripted-expert demonstrations.

Writes to --run-dir (default runs/<num_demos>_s<seed>):
  checkpoint.pt         EMA weights + normalizer + model kwargs
  config.yaml           fully resolved config
  loss.csv              step, loss (mean over log interval), lr, elapsed seconds
  train_summary.json    dataset seeds used, window count, timing, versions
  loss_curve.png
  train_state.pt        (while training) full resumable state, saved every
                        train.save_every steps; re-running the same command
                        resumes from it and it is deleted once training ends

Usage:
  python code/train.py --set data.num_demos=25 seed=1
  python code/train.py --set train.num_steps=300 --run-dir runs/debug      # quick check
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
from hw2.dataset import MinMaxNormalizer, load_demos, make_windows
from hw2.policy import EMA, DiffusionPolicy, atomic_torch_save, save_checkpoint


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


def _resume_key(cfg):
  """Config entries that must match for a saved training state to be resumed."""
  tc = cfg["train"]
  return (cfg["seed"], cfg["data"], cfg["policy"],
          {k: tc[k] for k in ["num_steps", "batch_size", "lr", "weight_decay", "warmup_steps", "ema_power"]})


def train(cfg, run_dir, stop_after=None):
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

  # Resume from the last periodic training state, if any (e.g. after a Colab disconnect;
  # on Colab runs/ lives on Google Drive). Everything that affects the remaining steps
  # is restored, including the minibatch/noise RNG, so the resumed run is the same run.
  state_path = run_dir / "train_state.pt"
  start_step, rows, elapsed_before = 0, [], 0.0
  if state_path.exists():
    st = torch.load(state_path, map_location=device, weights_only=False)
    if _resume_key(st["config"]) != _resume_key(cfg):
      raise RuntimeError(f"{state_path} was saved with a different config; delete it to start over.")
    policy.load_state_dict(st["policy"])
    ema.model.load_state_dict(st["ema"])
    ema.step = st["ema_step"]
    opt.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    gen.set_state(st["generator"].cpu())
    start_step, rows, elapsed_before = st["step"], st["rows"], st["elapsed_s"]
    print(f"resuming from step {start_step} ({state_path})", flush=True)

  def save_state(step):
    atomic_torch_save({
        "step": step, "config": cfg, "rows": rows, "elapsed_s": elapsed_before + time.time() - t0,
        "policy": policy.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "generator": gen.get_state(),
    }, state_path)

  save_every = tc.get("save_every", 0)
  running, t0 = [], time.time()
  policy.train()
  for step in range(start_step + 1, tc["num_steps"] + 1):
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
             "elapsed_s": elapsed_before + time.time() - t0}
      rows.append(row)
      running = []
      if step % (tc["log_every"] * 10) == 0 or step == tc["num_steps"]:
        rate = (step - start_step) / (time.time() - t0)
        print(f"step {step:6d} loss {row['loss']:.5f} lr {row['lr']:.2e} "
              f"{rate:.1f} it/s eta {(tc['num_steps'] - step) / rate / 60:.1f} min", flush=True)
    stopping = stop_after is not None and step - start_step >= stop_after and step < tc["num_steps"]
    if (save_every and step % save_every == 0 and step < tc["num_steps"]) or stopping:
      save_state(step)
    if stopping:
      print(f"stopped after {stop_after} steps at step {step}; state saved to {state_path}", flush=True)
      return None

  train_time = elapsed_before + time.time() - t0
  save_checkpoint(run_dir / "checkpoint.pt", ema.model, extra={"config": cfg, "step": tc["num_steps"]})
  state_path.unlink(missing_ok=True)  # final checkpoint supersedes the resume state
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
  parser.add_argument("--stop-after", type=int, default=None,
                      help="save the training state and exit after this many steps (resume later)")
  args = parser.parse_args()
  cfg = utils.load_config(args.config, args.overrides)
  run_dir = args.run_dir or f"runs/n{cfg['data']['num_demos']}_s{cfg['seed']}"
  if train(cfg, utils.resolve_path(run_dir), stop_after=args.stop_after) is None:
    sys.exit(3)  # stopped early on purpose: no final checkpoint yet


if __name__ == "__main__":
  main()
