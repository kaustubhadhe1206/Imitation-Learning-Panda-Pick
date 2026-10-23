"""Environment + pipeline smoke test. Run this first on a new machine/runtime.

Sections (all by default):
  env     versions, devices, PandaPickCube dims/ranges, jit + step timing
  expert  scripted expert on one batch of seeds: success rate, failure modes,
          phase durations (also the quick loop for tuning the expert:
          --set expert.v_approach=0.025 ...)
  render  one PNG frame + one expert video
  torch   Diffusion Policy training / inference speed -> time estimates

Ends with a compact JSON block to paste back.

Usage:
  python code/smoke_test.py
  python code/smoke_test.py --sections expert render --set expert.close_steps=10
"""

import argparse
import json
import time
import traceback

import numpy as np

import hw2  # noqa: F401
from hw2 import utils

REPORT = {}


def section(name):
  def deco(fn):
    def wrapped(*a, **k):
      print(f"\n===== {name} =====", flush=True)
      t0 = time.time()
      try:
        fn(*a, **k)
        REPORT.setdefault(name, {})["ok"] = True
      except Exception as e:  # report and continue with the other sections
        traceback.print_exc()
        REPORT.setdefault(name, {}).update({"ok": False, "error": repr(e)[:500]})
      REPORT[name]["seconds"] = round(time.time() - t0, 1)
    return wrapped
  return deco


@section("env")
def check_env(ctx):
  import jax
  info = utils.env_info()
  REPORT["env"] = {"packages": info["packages"], "gpu": info.get("gpu"), "python": info["python"],
                   "jax_devices": [str(d) for d in jax.devices()]}
  print(json.dumps(REPORT["env"], indent=2))

  from hw2.envs import BatchedEnv, env_config_dict
  t0 = time.time()
  benv = BatchedEnv()
  ctx["benv"] = benv
  m = benv.mj_model
  cfg = env_config_dict()
  dims = {
      "action_dim": benv.action_size, "nq": m.nq, "nv": m.nv, "nu": m.nu,
      "ctrl_range": m.actuator_ctrlrange.round(4).tolist(),
      "episode_length": benv.episode_length, "ctrl_dt": cfg["ctrl_dt"], "sim_dt": cfg["sim_dt"],
      "action_scale": benv.action_scale, "impl": cfg.get("impl"),
  }
  REPORT["env"].update(dims)
  print(json.dumps(dims, indent=2))

  B = ctx["batch"]
  seeds = np.arange(B)
  state = benv.reset(seeds)
  snap = benv.snapshot(state)
  t1 = time.time()
  state = benv.step(state, np.zeros((B, benv.action_size), np.float32))
  benv.snapshot(state)
  t2 = time.time()
  for _ in range(20):
    state = benv.step(state, np.zeros((B, benv.action_size), np.float32))
    benv.snapshot(state)
  t3 = time.time()
  REPORT["env"].update({
      "load_and_reset_compile_s": round(t1 - t0, 1), "step_compile_s": round(t2 - t1, 1),
      "batched_step_ms": round(1000 * (t3 - t2) / 20, 1), "batch": B,
      "obs_dim": int(snap["obs"].shape[-1]),
      "box_init_example": snap["box_pos"][0].round(3).tolist(),
      "target_example": snap["target_pos"][0].round(3).tolist(),
      "grip_init_example": snap["grip_pos"][0].round(3).tolist(),
  })
  print({k: REPORT["env"][k] for k in ["load_and_reset_compile_s", "step_compile_s", "batched_step_ms"]})


@section("expert")
def check_expert(ctx, overrides):
  from hw2.expert import PHASE_NAMES, ExpertController
  from hw2.rollout import classify_failures, rollout
  benv = ctx["benv"]
  cfg = utils.load_config("configs/collect.yaml", overrides)
  ctrl = ExpertController(benv.mj_model, benv.action_scale, cfg["expert"])
  seeds = np.arange(ctx["batch"])
  t0 = time.time()
  r = rollout(benv, seeds, ctrl, benv.episode_length, success_threshold=cfg["success_threshold"])
  ctx["expert_result"] = r
  modes = classify_failures(r)
  succ = r["success"]
  phase = r["phase"]
  rep = {
      "success_rate": float(succ.mean()),
      "failure_modes": {m: int((modes == m).sum()) for m in np.unique(modes)},
      "final_dist_cm": {"median": float(100 * np.median(r["dist"][:, -1])),
                        "max_success": float(100 * r["dist"][succ, -1].max()) if succ.any() else None},
      "first_success_step_mean": float(r["first_success_step"][succ].mean()) if succ.any() else None,
      "phase_steps_mean": {n: float((phase == i).sum(axis=1).mean()) for i, n in enumerate(PHASE_NAMES)},
      "final_phase_counts": {PHASE_NAMES[p]: int((phase[:, -1] == p).sum()) for p in np.unique(phase[:, -1])},
      "regrasp_episodes": int((np.diff(phase.astype(int), axis=1) < 0).any(axis=1).sum()),
      "reached_box_flag": float((r["reached_box"][:, -1] > 0.5).mean()),
      "action_saturation_frac": float((np.abs(r["actions"]) > 0.999).mean()),
      "min_grip_box_cm_median": float(100 * np.median(np.linalg.norm(r["grip_pos"] - r["box_pos"], axis=-1).min(1))),
      "rollout_s": round(time.time() - t0, 1),
  }
  fails = np.flatnonzero(~succ)[:5]
  rep["failed_seeds"] = [int(s) for s in r["seeds"][fails]]
  rep["failed_final_dist_cm"] = [round(float(100 * r["dist"][i, -1]), 1) for i in fails]
  rep["failed_final_phase"] = [PHASE_NAMES[int(phase[i, -1])] for i in fails]
  REPORT["expert"] = rep
  print(json.dumps(rep, indent=2))


@section("render")
def check_render(ctx):
  from hw2.render import TrajectoryRenderer, annotate, write_video
  benv = ctx["benv"]
  out = utils.resolve_path("results/smoke")
  out.mkdir(parents=True, exist_ok=True)
  rend = TrajectoryRenderer(benv.mj_model, height=240, width=320)
  r = ctx.get("expert_result")
  if r is None:
    state = benv.reset(np.arange(ctx["batch"]))
    s = benv.snapshot(state)
    frames = rend.render(s["qpos"][:1], s["target_pos"][0])
  else:
    i = int(np.argmax(r["success"])) if r["success"].any() else 0
    frames = annotate(rend.render(r["qpos"][i], r["target_pos"][i], every=2),
                      f"smoke test expert seed {int(r['seeds'][i])}")
    REPORT["render"] = {"video": str(write_video(utils.resolve_path("videos/smoke/expert_smoke.mp4"), frames, 25))}
    j = np.flatnonzero(~r["success"])
    if len(j):
      ff = annotate(rend.render(r["qpos"][j[0]], r["target_pos"][j[0]], every=2),
                    f"smoke test expert FAILURE seed {int(r['seeds'][j[0]])}")
      REPORT["render"]["failure_video"] = str(write_video(
          utils.resolve_path("videos/smoke/expert_smoke_failure.mp4"), ff, 25))
  import imageio.v2 as imageio
  imageio.imwrite(out / "render_test.png", frames[len(frames) // 2])
  REPORT["render"] = {**REPORT.get("render", {}), "png": str(out / "render_test.png"),
                      "frame_mean_intensity": float(frames[0].mean())}
  rend.close()
  print(REPORT["render"])


@section("torch")
def check_torch(ctx):
  import torch
  from hw2.policy import DiffusionPolicy
  cfg = utils.load_config("configs/dp_base.yaml")
  pc, tc = cfg["policy"], cfg["train"]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  obs_dim = REPORT.get("env", {}).get("obs_dim", 66)
  policy = DiffusionPolicy(obs_dim=obs_dim, act_dim=8, **pc).to(device)
  opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
  B = tc["batch_size"]
  nobs = torch.rand(B, pc["obs_horizon"], obs_dim, device=device) * 2 - 1
  nact = torch.rand(B, pc["pred_horizon"], 8, device=device) * 2 - 1

  def train_steps(n):
    for _ in range(n):
      loss = policy.compute_loss(nobs, nact)
      opt.zero_grad()
      loss.backward()
      opt.step()
    if device.type == "cuda":
      torch.cuda.synchronize()

  train_steps(5)
  t0 = time.time()
  train_steps(50)
  sec_per_step = (time.time() - t0) / 50
  policy.eval()
  obs = torch.rand(cfg["eval"]["batch_size"], pc["obs_horizon"], obs_dim, device=device)
  policy.predict_action(obs)
  if device.type == "cuda":
    torch.cuda.synchronize()
  t0 = time.time()
  policy.predict_action(obs)
  if device.type == "cuda":
    torch.cuda.synchronize()
  infer = time.time() - t0
  n_chunks = int(np.ceil(150 / pc["action_horizon"]))
  REPORT["torch"] = {
      "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
      "params_M": round(sum(p.numel() for p in policy.parameters()) / 1e6, 2),
      "train_s_per_step": round(sec_per_step, 4),
      "est_train_min_per_run": round(sec_per_step * tc["num_steps"] / 60, 1),
      "infer_s_per_chunk_batch": round(infer, 3),
      "est_eval_policy_min": round(infer * n_chunks * cfg["eval"]["num_episodes"] / cfg["eval"]["batch_size"] / 60, 2),
      "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
  }
  print(REPORT["torch"])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sections", nargs="*", default=["env", "expert", "render", "torch"])
  parser.add_argument("--batch", type=int, default=50, help="envs in parallel (match collect.yaml)")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                      help="overrides for configs/collect.yaml (expert tuning)")
  args = parser.parse_args()
  ctx = {"batch": args.batch}
  if {"env", "expert", "render"} & set(args.sections):
    check_env(ctx)
  if "expert" in args.sections and "benv" in ctx:
    check_expert(ctx, args.overrides)
  if "render" in args.sections and "benv" in ctx:
    check_render(ctx)
  if "torch" in args.sections:
    check_torch(ctx)
  print("\n==== SMOKE TEST REPORT (paste this back) ====")
  print(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
  main()
