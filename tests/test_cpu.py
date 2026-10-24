"""Tests that run on a normal CPU (no MuJoCo or JAX needed): dataset, diffusion,
policy, episode bookkeeping, training resume, analysis and figures on fake data.

  python -m pytest tests/test_cpu.py -q        (or: python tests/test_cpu.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import hw2  # noqa: E402,F401
from hw2 import utils  # noqa: E402
from hw2.dataset import MinMaxNormalizer, get_window_time_indices, make_training_windows  # noqa: E402
from hw2.diffusion import ConditionalUnet1D, NoiseScheduler  # noqa: E402
from hw2.policy import (DiffusionPolicy, DiffusionPolicyController,  # noqa: E402
                        ExponentialMovingAverage, load_checkpoint, save_checkpoint)
from hw2.rollout import (ReplayActionsController, make_episode_rows, run_episodes,  # noqa: E402
                         summarize_episodes)

OBS_DIM, ACTION_DIM, NUM_STEPS = 66, 8, 150
TINY_POLICY = dict(obs_horizon=2, pred_horizon=16, action_horizon=8, down_dims=(16, 32, 64),
                   diffusion_step_embed_dim=16, num_train_timesteps=20, num_inference_steps=20)


class FakeBatchedEnv:
  """Stand-in for the real env: the box moves with the first 3 action numbers (1 cm per unit)."""

  episode_length = NUM_STEPS

  def reset(self, seeds):
    rng = np.random.default_rng(int(seeds[0]))
    batch_size = len(seeds)
    box = np.c_[rng.uniform(0.3, 0.7, batch_size), rng.uniform(-0.2, 0.2, batch_size), np.full(batch_size, 0.03)]
    target = box + np.c_[np.zeros((batch_size, 2)), np.full(batch_size, 0.25)]
    return {"box": box, "target": target}

  def step(self, state, actions):
    return {**state, "box": state["box"] + 0.01 * np.asarray(actions)[:, :3]}

  def to_numpy(self, state):
    batch_size = len(state["box"])
    obs = np.zeros((batch_size, OBS_DIM), np.float32)
    obs[:, :3], obs[:, 3:6] = state["box"], state["target"]
    return {"obs": obs, "qpos": np.zeros((batch_size, 16), np.float32),
            "ctrl": np.zeros((batch_size, 8), np.float32), "grip_pos": state["box"].copy(),
            "box_pos": state["box"].copy(), "target_pos": state["target"],
            "reward": np.zeros(batch_size, np.float32), "done": np.zeros(batch_size, np.float32),
            "reached_box": np.ones(batch_size, np.float32)}


class MoveToTargetController:
  """Simple fake expert for the fake env: push the box toward the target."""

  def reset(self, env_state):
    self.current_phases = np.zeros(len(env_state["obs"]), np.int8)

  def get_action(self, env_state, step):
    actions = np.zeros((len(env_state["obs"]), ACTION_DIM), np.float32)
    actions[:, :3] = np.clip((env_state["target_pos"] - env_state["box_pos"]) * 20, -1, 1)
    return actions


def test_window_time_indices():
  time_indices = get_window_time_indices(10, obs_horizon=2, pred_horizon=4, action_horizon=2)
  assert time_indices.shape == (10 - 4 + 2 + 2 - 1, 4)
  assert time_indices[0].tolist() == [0, 0, 1, 2]     # first frame repeated at the start
  assert time_indices[-1].tolist() == [7, 8, 9, 9]    # last frame repeated at the end
  obs = np.random.rand(3, 10, 5).astype(np.float32)
  actions = np.random.rand(3, 10, 2).astype(np.float32)
  obs_windows, action_windows = make_training_windows(obs, actions, 2, 4, 2)
  assert obs_windows.shape == (3 * 9, 2, 5) and action_windows.shape == (3 * 9, 4, 2)
  np.testing.assert_array_equal(obs_windows[1], obs[0, [0, 1]])
  np.testing.assert_array_equal(action_windows[1], actions[0, [0, 1, 2, 3]])


def test_normalizer():
  data = np.random.randn(100, 7).astype(np.float32)
  data[:, 3] = 2.5   # a dimension that never changes
  normalizer = MinMaxNormalizer.fit(data)
  normalized = normalizer.normalize(data)
  changing_dims = [0, 1, 2, 4, 5, 6]
  assert np.allclose(normalized.min(0)[changing_dims], -1, atol=1e-5)
  assert np.allclose(normalized.max(0)[changing_dims], 1, atol=1e-5)
  assert np.allclose(normalized[:, 3], 0)
  assert np.allclose(normalizer.unnormalize(normalized), data, atol=1e-5)


def test_unet_output_shape():
  network = ConditionalUnet1D(ACTION_DIM, OBS_DIM * 2, diffusion_step_embed_dim=16, down_dims=(16, 32, 64))
  noisy_actions = torch.randn(4, 16, ACTION_DIM)
  output = network(noisy_actions, torch.randint(0, 100, (4,)), torch.randn(4, OBS_DIM * 2))
  assert output.shape == noisy_actions.shape
  output = network(noisy_actions, 5, torch.randn(4, OBS_DIM * 2))
  assert output.shape == noisy_actions.shape


def test_sampler_recovers_point_with_perfect_noise_prediction():
  """If the 'network' predicts the noise perfectly for a single target point, sampling must return that point."""
  torch.manual_seed(0)
  scheduler = NoiseScheduler(100)
  target_point = torch.rand(3, 16, ACTION_DIM) * 1.6 - 0.8

  def perfect_noise_prediction(noisy, timesteps):
    alpha_bar = scheduler.alpha_bars.float()[timesteps].reshape(-1, 1, 1)
    return (noisy - alpha_bar.sqrt() * target_point) / (1 - alpha_bar).sqrt()

  for num_inference_steps in (100, 10):   # full DDPM and DDIM with skipped steps
    sample = scheduler.generate(perfect_noise_prediction, target_point.shape, "cpu",
                                num_inference_steps=num_inference_steps)
    assert torch.allclose(sample, target_point, atol=1e-3), (num_inference_steps,
                                                            (sample - target_point).abs().max())
  noisy = scheduler.add_noise(target_point, torch.zeros_like(target_point), torch.zeros(3, dtype=torch.long))
  assert torch.allclose(noisy, target_point * scheduler.alpha_bars[0].float().sqrt())


def test_policy_learns_and_checkpoint_roundtrip():
  torch.manual_seed(0)
  obs = np.random.rand(20, 40, OBS_DIM).astype(np.float32)
  actions = np.tanh(obs[..., :ACTION_DIM] * 3 - 1.5).astype(np.float32)
  obs_windows, action_windows = make_training_windows(obs, actions, 2, 16, 8)
  obs_normalizer, action_normalizer = MinMaxNormalizer.fit(obs), MinMaxNormalizer.fit(actions)
  normalized_obs = torch.from_numpy(obs_normalizer.normalize(obs_windows))
  normalized_actions = torch.from_numpy(action_normalizer.normalize(action_windows))
  policy = DiffusionPolicy(OBS_DIM, ACTION_DIM, **TINY_POLICY)
  policy.set_normalizers(obs_normalizer, action_normalizer)
  ema = ExponentialMovingAverage(policy)
  optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
  losses = []
  for _ in range(100):
    batch_ids = torch.randint(0, len(normalized_obs), (32,))
    loss = policy.compute_loss(normalized_obs[batch_ids], normalized_actions[batch_ids])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(policy)
    losses.append(loss.item())
  # Only a quick check (fixed seed, tiny network, 100 steps): the loss should go down.
  assert np.mean(losses[-20:]) < 0.95 * np.mean(losses[:20]), (losses[:5], losses[-5:])
  assert torch.equal(ema.averaged_model.action_norm_scale, policy.action_norm_scale)

  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "checkpoint.pt"
    save_checkpoint(path, ema.averaged_model, extra={"config": {"x": 1}})
    loaded_policy, checkpoint = load_checkpoint(path, "cpu")
    assert checkpoint["config"] == {"x": 1}
    obs_history = torch.from_numpy(obs[:3, :2])
    rng_a, rng_b = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    actions_a = ema.averaged_model.predict_action_sequence(obs_history, generator=rng_a)
    actions_b = loaded_policy.predict_action_sequence(obs_history, generator=rng_b)
    assert actions_a.shape == (3, 16, ACTION_DIM)
    assert torch.allclose(actions_a, actions_b, atol=1e-5)


def test_episodes_success_and_replay():
  env = FakeBatchedEnv()
  seeds = np.arange(6)
  results = run_episodes(env, seeds, MoveToTargetController(), NUM_STEPS)
  assert results["obs"].shape == (6, NUM_STEPS, OBS_DIM)
  assert results["actions"].shape == (6, NUM_STEPS, ACTION_DIM)
  assert results["qpos"].shape == (6, NUM_STEPS + 1, 16) and results["dist"].shape == (6, NUM_STEPS + 1)
  assert results["phase"].shape == (6, NUM_STEPS)
  assert results["success"].all()
  assert (results["first_success_step"] > 0).all()
  # obs[t] must be the observation from BEFORE action t
  np.testing.assert_allclose(results["obs"][:, 1, :3], results["box_pos"][:, 1], atol=1e-6)
  replay = run_episodes(env, seeds, ReplayActionsController(results["actions"]), NUM_STEPS)
  np.testing.assert_allclose(replay["obs"], results["obs"])
  # with noise: executed actions differ from the wanted ones and stay in [-1, 1]
  noisy = run_episodes(env, seeds, MoveToTargetController(), NUM_STEPS, noise_std=0.5,
                       noise_rng=np.random.default_rng(0))
  assert np.abs(noisy["actions"]).max() <= 1.0
  assert not np.allclose(noisy["actions"], noisy["actions_clean"])
  rows = make_episode_rows(results, chunk_size=8)
  summary = summarize_episodes(results, chunk_size=8)
  assert summary["success_rate"] == 1.0 and rows[0]["failure_mode"] == "success"
  assert {"jerk_boundary", "jerk_within"} <= set(summary)


def test_failure_labels():
  env = FakeBatchedEnv()
  do_nothing = ReplayActionsController(np.zeros((4, NUM_STEPS, ACTION_DIM), np.float32))
  summary = summarize_episodes(run_episodes(env, np.arange(4), do_nothing, NUM_STEPS))
  assert summary["success_rate"] == 0.0
  assert summary["failure_modes"]["grasp_failed"] == 4   # gripper at the box, but the box never lifted


def test_policy_controller_replans_every_action_horizon():
  env = FakeBatchedEnv()
  policy = DiffusionPolicy(OBS_DIM, ACTION_DIM, **TINY_POLICY).eval()
  controller = DiffusionPolicyController(policy, "cpu", seed=0)
  results = run_episodes(env, np.arange(3), controller, 40)
  assert controller.num_policy_calls == 5   # ceil(40 / 8)
  assert np.abs(results["actions"]).max() <= 1.0


def test_config_overrides():
  config = utils.load_config("configs/dp_base.yaml",
                             ["seed=3", "train.num_steps=10", "policy.down_dims=[8,16]"])
  assert config["seed"] == 3 and config["train"]["num_steps"] == 10
  assert config["policy"]["down_dims"] == [8, 16]
  sweep = utils.load_config("configs/sweep.yaml")
  assert [condition["num_demos"] for condition in sweep["conditions"]] == [100, 50, 25]
  ci_low, ci_high = utils.wilson_confidence_interval(50, 100)
  assert 0.40 < ci_low < 0.41 and 0.59 < ci_high < 0.60


def test_training_resume_gives_same_model():
  """Stopping and continuing training must give exactly the same model as training in one go."""
  import train as train_script
  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    rng = np.random.default_rng(0)
    obs = rng.random((4, 30, OBS_DIM), dtype=np.float32)
    np.savez(tmp / "demos.npz", obs=obs, actions=np.tanh(obs[..., :ACTION_DIM]), seeds=np.arange(4),
             success=np.ones(4, bool))
    config = utils.load_config("configs/dp_base.yaml", [
        f"data.path={(tmp / 'demos.npz').as_posix()}", "data.num_demos=4", "policy.down_dims=[8,16,32]",
        "policy.diffusion_step_embed_dim=8", "policy.num_train_timesteps=10", "train.num_steps=30",
        "train.batch_size=8", "train.warmup_steps=5", "train.log_every=5", "train.save_every=10"])
    train_script.train(config, tmp / "in_one_go")
    assert train_script.train(config, tmp / "resumed", stop_after=15) is None   # pretend Colab died at step 15
    assert (tmp / "resumed" / "train_state.pt").exists()
    train_script.train(config, tmp / "resumed")                                 # continue to the end
    assert not (tmp / "resumed" / "train_state.pt").exists()
    weights_a = torch.load(tmp / "in_one_go" / "checkpoint.pt", weights_only=False)["state_dict"]
    weights_b = torch.load(tmp / "resumed" / "checkpoint.pt", weights_only=False)["state_dict"]
    for name in weights_a:
      assert torch.equal(weights_a[name], weights_b[name]), name
    loss_a = (tmp / "in_one_go" / "loss.csv").read_text().splitlines()
    loss_b = (tmp / "resumed" / "loss.csv").read_text().splitlines()
    assert [line.split(",")[:2] for line in loss_a] == [line.split(",")[:2] for line in loss_b]
    # continuing with a different config must be refused
    different_config = dict(config, seed=1)
    (tmp / "resumed" / "checkpoint.pt").unlink()
    train_script.train(different_config, tmp / "resumed", stop_after=5)
    try:
      train_script.train(config, tmp / "resumed")
      raise AssertionError("continuing with a different config should fail")
    except RuntimeError:
      pass


def test_analysis_and_demo_figures():
  import analyze
  import inspect_demos
  from hw2 import plotting
  env = FakeBatchedEnv()
  results = run_episodes(env, np.arange(12), MoveToTargetController(), NUM_STEPS)
  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # demo figures on fake data
    demos = dict(results)
    demos["meta"] = np.array('{"name": "fake", "noise_std": 0.0, "teacher_success_rate": 1.0}')
    demos["success"] = demos["success"].copy()
    demos["success"][0] = False
    stats = inspect_demos.compute_dataset_stats(demos)
    assert stats["num_trajectories"] == 12 and stats["num_successful"] == 11
    plotting.use_report_style()
    for plot_function in [inspect_demos.plot_initial_conditions, inspect_demos.plot_trajectories,
                          inspect_demos.plot_actions, inspect_demos.plot_phase_durations]:
      assert plot_function(demos, tmp / f"{plot_function.__name__}.png").exists()
    # sweep analysis on fake results
    analyze.RESULTS_DIR = tmp
    for condition_name, rate in [("n100", 0.9), ("n50", 0.7), ("n25", 0.4)]:
      for seed in (0, 1):
        run_folder = tmp / "runs" / f"{condition_name}_s{seed}"
        summary = summarize_episodes(results, chunk_size=8)
        summary.update(success_rate=rate + 0.02 * seed, num_success=int(100 * rate), num_episodes=100)
        utils.save_json(summary, run_folder / "eval_summary.json")
        utils.save_json({"num_demos": 1}, run_folder / "train_summary.json")
        (run_folder / "loss.csv").write_text("step,loss,lr,elapsed_s\n100,0.5,1e-4,1\n200,0.2,1e-4,2\n")
    utils.save_json(summarize_episodes(results, chunk_size=8), tmp / "teacher_eval" / "eval_summary.json")
    utils.save_json({"teacher_success_rate": 0.95}, tmp / "demos" / "collection_baseline.json")
    sys.argv = ["analyze.py"]
    analyze.main()
    for file_name in ["summary_table.md", "fig_success_vs_demos.png", "fig_success_vs_threshold.png",
                      "fig_loss_curves.png", "fig_failure_modes.png", "fig_smoothness.png"]:
      assert (tmp / file_name).exists(), file_name
    assert "| n25 | 25 |" in (tmp / "summary_table.md").read_text()


if __name__ == "__main__":
  import time
  all_tests = [function for name, function in dict(globals()).items() if name.startswith("test_")]
  for test in all_tests:
    start_time = time.time()
    test()
    print(f"ok {test.__name__} ({time.time() - start_time:.1f}s)")
