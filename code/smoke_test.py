"""Quick check that everything works on a new machine. Run this first.

Checks (all of them by default):
  env      versions, GPU, PandaPickCube sizes and ranges, compile + step time
  expert   scripted expert on one batch of seeds: success rate, failure types,
           time per phase. Also the quick way to tune the expert, e.g.
           --set expert.approach_speed=0.025
  render   one PNG image + one expert video
  torch    Diffusion Policy training / sampling speed -> time estimates

At the end it prints a short JSON report to paste back.

Usage:
  python code/smoke_test.py
  python code/smoke_test.py --checks expert render --set expert.gripper_close_steps=10
"""

import argparse
import json
import time
import traceback

import numpy as np

import hw2  # noqa: F401
from hw2 import utils

REPORT = {}


def report_section(name):
  """Decorator: run a check, store whether it worked (and how long it took) in REPORT, never crash."""
  def decorator(check_function):
    def wrapper(*args, **kwargs):
      print(f"\n===== {name} =====", flush=True)
      start_time = time.time()
      try:
        check_function(*args, **kwargs)
        REPORT.setdefault(name, {})["ok"] = True
      except Exception as error:   # print the error and keep going with the other checks
        traceback.print_exc()
        REPORT.setdefault(name, {}).update({"ok": False, "error": repr(error)[:500]})
      REPORT[name]["seconds"] = round(time.time() - start_time, 1)
    return wrapper
  return decorator


@report_section("env")
def check_environment(shared):
  import jax
  system_info = utils.get_system_info()
  REPORT["env"] = {"packages": system_info["packages"], "gpu": system_info.get("gpu"),
                   "python": system_info["python"], "jax_devices": [str(d) for d in jax.devices()]}
  print(json.dumps(REPORT["env"], indent=2))

  from hw2.envs import BatchedPickCubeEnv, get_env_config
  start_time = time.time()
  batched_env = BatchedPickCubeEnv()
  shared["batched_env"] = batched_env
  model = batched_env.mj_model
  env_config = get_env_config()
  sizes = {
      "action_dim": batched_env.action_size, "nq": model.nq, "nv": model.nv, "nu": model.nu,
      "ctrl_range": model.actuator_ctrlrange.round(4).tolist(),
      "episode_length": batched_env.episode_length, "ctrl_dt": env_config["ctrl_dt"],
      "sim_dt": env_config["sim_dt"], "action_scale": batched_env.action_scale, "impl": env_config.get("impl"),
  }
  REPORT["env"].update(sizes)
  print(json.dumps(sizes, indent=2))

  batch_size = shared["batch_size"]
  zero_actions = np.zeros((batch_size, batched_env.action_size), np.float32)
  sim_state = batched_env.reset(np.arange(batch_size))
  env_state = batched_env.to_numpy(sim_state)
  reset_done = time.time()
  sim_state = batched_env.step(sim_state, zero_actions)
  batched_env.to_numpy(sim_state)
  first_step_done = time.time()
  for _ in range(20):
    sim_state = batched_env.step(sim_state, zero_actions)
    batched_env.to_numpy(sim_state)
  twenty_steps_done = time.time()
  REPORT["env"].update({
      "load_and_reset_compile_s": round(reset_done - start_time, 1),
      "step_compile_s": round(first_step_done - reset_done, 1),
      "batched_step_ms": round(1000 * (twenty_steps_done - first_step_done) / 20, 1),
      "batch": batch_size,
      "obs_dim": int(env_state["obs"].shape[-1]),
      "box_init_example": env_state["box_pos"][0].round(3).tolist(),
      "target_example": env_state["target_pos"][0].round(3).tolist(),
      "grip_init_example": env_state["grip_pos"][0].round(3).tolist(),
  })
  print({key: REPORT["env"][key] for key in ["load_and_reset_compile_s", "step_compile_s", "batched_step_ms"]})


@report_section("expert")
def check_expert(shared, overrides):
  from hw2.expert import PHASE_NAMES, BatchedExpertController
  from hw2.rollout import label_failure_modes, run_episodes
  batched_env = shared["batched_env"]
  config = utils.load_config("configs/collect.yaml", overrides)
  expert = BatchedExpertController(batched_env.mj_model, batched_env.action_scale, config["expert"])
  start_time = time.time()
  results = run_episodes(batched_env, np.arange(shared["batch_size"]), expert, batched_env.episode_length,
                         success_threshold=config["success_threshold"])
  shared["expert_results"] = results
  failure_modes = label_failure_modes(results)
  successful = results["success"]
  phases = results["phase"]
  gripper_to_box = np.linalg.norm(results["grip_pos"] - results["box_pos"], axis=-1)
  expert_report = {
      "success_rate": float(successful.mean()),
      "failure_modes": {mode: int((failure_modes == mode).sum()) for mode in np.unique(failure_modes)},
      "final_dist_cm": {"median": float(100 * np.median(results["dist"][:, -1])),
                        "max_success": float(100 * results["dist"][successful, -1].max()) if successful.any() else None},
      "first_success_step_mean": float(results["first_success_step"][successful].mean()) if successful.any() else None,
      "phase_steps_mean": {name: float((phases == i).sum(axis=1).mean()) for i, name in enumerate(PHASE_NAMES)},
      "final_phase_counts": {PHASE_NAMES[p]: int((phases[:, -1] == p).sum()) for p in np.unique(phases[:, -1])},
      "regrasp_episodes": int((np.diff(phases.astype(int), axis=1) < 0).any(axis=1).sum()),
      "reached_box_flag": float((results["reached_box"][:, -1] > 0.5).mean()),
      "action_saturation_frac": float((np.abs(results["actions"]) > 0.999).mean()),
      "min_grip_box_cm_median": float(100 * np.median(gripper_to_box.min(1))),
      "rollout_s": round(time.time() - start_time, 1),
  }
  failed_ids = np.flatnonzero(~successful)[:5]
  expert_report["failed_seeds"] = [int(seed) for seed in results["seeds"][failed_ids]]
  expert_report["failed_final_dist_cm"] = [round(float(100 * results["dist"][i, -1]), 1) for i in failed_ids]
  expert_report["failed_final_phase"] = [PHASE_NAMES[int(phases[i, -1])] for i in failed_ids]
  REPORT["expert"] = expert_report
  print(json.dumps(expert_report, indent=2))


@report_section("render")
def check_rendering(shared):
  import imageio.v2 as imageio
  from hw2.render import TrajectoryRenderer, add_caption, save_video
  batched_env = shared["batched_env"]
  out_dir = utils.repo_path("results/smoke")
  out_dir.mkdir(parents=True, exist_ok=True)
  renderer = TrajectoryRenderer(batched_env.mj_model, height=240, width=320)
  results = shared.get("expert_results")
  render_report = {}
  if results is None:   # expert check was skipped: just render the first state
    env_state = batched_env.to_numpy(batched_env.reset(np.arange(shared["batch_size"])))
    frames = renderer.render_frames(env_state["qpos"][:1], env_state["target_pos"][0])
  else:
    i = int(np.argmax(results["success"])) if results["success"].any() else 0
    frames = add_caption(renderer.render_frames(results["qpos"][i], results["target_pos"][i], every_nth=2),
                         f"smoke test expert seed {int(results['seeds'][i])}")
    render_report["video"] = str(save_video(utils.repo_path("videos/smoke/expert_smoke.mp4"), frames, 25))
    failed_ids = np.flatnonzero(~results["success"])
    if len(failed_ids):
      j = failed_ids[0]
      failure_frames = add_caption(renderer.render_frames(results["qpos"][j], results["target_pos"][j], every_nth=2),
                                   f"smoke test expert FAILURE seed {int(results['seeds'][j])}")
      render_report["failure_video"] = str(save_video(
          utils.repo_path("videos/smoke/expert_smoke_failure.mp4"), failure_frames, 25))
  imageio.imwrite(out_dir / "render_test.png", frames[len(frames) // 2])
  render_report.update({"png": str(out_dir / "render_test.png"),
                        "frame_mean_intensity": float(frames[0].mean())})   # ~0 would mean a black image
  REPORT["render"] = render_report
  renderer.close()
  print(render_report)


@report_section("torch")
def check_training_speed(shared):
  import torch
  from hw2.policy import DiffusionPolicy
  config = utils.load_config("configs/dp_base.yaml")
  policy_config, train_config = config["policy"], config["train"]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  obs_dim = REPORT.get("env", {}).get("obs_dim", 66)
  policy = DiffusionPolicy(obs_dim=obs_dim, action_dim=8, **policy_config).to(device)
  optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
  batch_size = train_config["batch_size"]
  fake_obs = torch.rand(batch_size, policy_config["obs_horizon"], obs_dim, device=device) * 2 - 1
  fake_actions = torch.rand(batch_size, policy_config["pred_horizon"], 8, device=device) * 2 - 1

  def wait_for_gpu():
    if device.type == "cuda":
      torch.cuda.synchronize()

  def do_training_steps(num_steps):
    for _ in range(num_steps):
      loss = policy.compute_loss(fake_obs, fake_actions)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    wait_for_gpu()

  do_training_steps(5)   # warm up
  start_time = time.time()
  do_training_steps(50)
  seconds_per_step = (time.time() - start_time) / 50

  policy.eval()
  eval_obs = torch.rand(config["eval"]["batch_size"], policy_config["obs_horizon"], obs_dim, device=device)
  policy.predict_action_sequence(eval_obs)   # warm up
  wait_for_gpu()
  start_time = time.time()
  policy.predict_action_sequence(eval_obs)
  wait_for_gpu()
  seconds_per_sample = time.time() - start_time
  policy_calls_per_episode = int(np.ceil(150 / policy_config["action_horizon"]))
  num_eval_batches = config["eval"]["num_episodes"] / config["eval"]["batch_size"]
  REPORT["torch"] = {
      "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
      "params_M": round(sum(param.numel() for param in policy.parameters()) / 1e6, 2),
      "train_s_per_step": round(seconds_per_step, 4),
      "est_train_min_per_run": round(seconds_per_step * train_config["num_steps"] / 60, 1),
      "infer_s_per_chunk_batch": round(seconds_per_sample, 3),
      "est_eval_policy_min": round(seconds_per_sample * policy_calls_per_episode * num_eval_batches / 60, 2),
      "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
  }
  print(REPORT["torch"])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checks", nargs="*", default=["env", "expert", "render", "torch"])
  parser.add_argument("--batch", type=int, default=50, help="episodes in parallel (same as collect.yaml)")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                      help="changes to configs/collect.yaml (for tuning the expert)")
  args = parser.parse_args()
  shared = {"batch_size": args.batch}
  if {"env", "expert", "render"} & set(args.checks):
    check_environment(shared)
  if "expert" in args.checks and "batched_env" in shared:
    check_expert(shared, args.overrides)
  if "render" in args.checks and "batched_env" in shared:
    check_rendering(shared)
  if "torch" in args.checks:
    check_training_speed(shared)
  print("\n==== SMOKE TEST REPORT (paste this back) ====")
  print(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
  main()
