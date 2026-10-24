"""CPU-only unit tests (no MuJoCo / JAX needed): dataset, diffusion, policy,
rollout bookkeeping, analysis and demo figures on synthetic data.

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
from hw2.dataset import MinMaxNormalizer, make_windows, window_indices  # noqa: E402
from hw2.diffusion import ConditionalUnet1D, NoiseScheduler  # noqa: E402
from hw2.policy import (EMA, DiffusionPolicy, DiffusionPolicyController,  # noqa: E402
                        load_checkpoint, save_checkpoint)
from hw2.rollout import (ReplayController, episode_table, rollout,  # noqa: E402
                         summarize)

OBS_DIM, ACT_DIM, T = 66, 8, 150
SMALL = dict(obs_horizon=2, pred_horizon=16, action_horizon=8, down_dims=(16, 32, 64),
             diffusion_step_embed_dim=16, num_train_timesteps=20, num_inference_steps=20)


class FakeBatchedEnv:
  """Box follows the first 3 action dims (1 cm per unit action); obs = [box, target, 0...]."""

  episode_length = T

  def reset(self, seeds):
    rng = np.random.default_rng(int(seeds[0]))
    B = len(seeds)
    box = np.c_[rng.uniform(0.3, 0.7, B), rng.uniform(-0.2, 0.2, B), np.full(B, 0.03)]
    tgt = box + np.c_[np.zeros((B, 2)), np.full(B, 0.25)]
    return {"box": box, "tgt": tgt, "t": 0}

  def step(self, state, actions):
    return {**state, "box": state["box"] + 0.01 * np.asarray(actions)[:, :3], "t": state["t"] + 1}

  def snapshot(self, state):
    B = len(state["box"])
    obs = np.zeros((B, OBS_DIM), np.float32)
    obs[:, :3], obs[:, 3:6] = state["box"], state["tgt"]
    return {"obs": obs, "qpos": np.zeros((B, 16), np.float32), "ctrl": np.zeros((B, 8), np.float32),
            "grip_pos": state["box"].copy(), "box_pos": state["box"].copy(), "target_pos": state["tgt"],
            "reward": np.zeros(B, np.float32), "done": np.zeros(B, np.float32),
            "reached_box": np.ones(B, np.float32)}


class GoToTarget:
  """Proportional 'expert' for the fake env."""

  def reset(self, snap):
    self.phase = np.zeros(len(snap["obs"]), np.int8)

  def act(self, snap, t):
    a = np.zeros((len(snap["obs"]), ACT_DIM), np.float32)
    a[:, :3] = np.clip((snap["target_pos"] - snap["box_pos"]) * 20, -1, 1)
    return a


def test_window_indices():
  ix = window_indices(10, obs_horizon=2, pred_horizon=4, action_horizon=2)
  assert ix.shape == (10 - 4 + 2 + 2 - 1, 4)
  assert ix[0].tolist() == [0, 0, 1, 2]     # first obs repeated (edge pad before)
  assert ix[-1].tolist() == [7, 8, 9, 9]    # last frame repeated (edge pad after)
  obs = np.random.rand(3, 10, 5).astype(np.float32)
  act = np.random.rand(3, 10, 2).astype(np.float32)
  ow, aw = make_windows(obs, act, 2, 4, 2)
  assert ow.shape == (3 * 9, 2, 5) and aw.shape == (3 * 9, 4, 2)
  np.testing.assert_array_equal(ow[1], obs[0, [0, 1]])
  np.testing.assert_array_equal(aw[1], act[0, [0, 1, 2, 3]])


def test_normalizer():
  x = np.random.randn(100, 7).astype(np.float32)
  x[:, 3] = 2.5  # constant dimension
  n = MinMaxNormalizer.fit(x)
  y = n.normalize(x)
  assert np.allclose(y.min(0)[[0, 1, 2, 4, 5, 6]], -1, atol=1e-5)
  assert np.allclose(y.max(0)[[0, 1, 2, 4, 5, 6]], 1, atol=1e-5)
  assert np.allclose(y[:, 3], 0)
  assert np.allclose(n.unnormalize(y), x, atol=1e-5)


def test_unet_shapes():
  net = ConditionalUnet1D(ACT_DIM, OBS_DIM * 2, diffusion_step_embed_dim=16, down_dims=(16, 32, 64))
  x = torch.randn(4, 16, ACT_DIM)
  out = net(x, torch.randint(0, 100, (4,)), torch.randn(4, OBS_DIM * 2))
  assert out.shape == x.shape
  out = net(x, 5, torch.randn(4, OBS_DIM * 2))
  assert out.shape == x.shape


def test_scheduler_oracle_recovers_point():
  """With the exact noise predictor for a point mass x*, sampling must return x*."""
  torch.manual_seed(0)
  sch = NoiseScheduler(100)
  x_star = torch.rand(3, 16, ACT_DIM) * 1.6 - 0.8

  def oracle(x, t):
    ab = sch.alphas_cumprod.float()[t].reshape(-1, 1, 1)
    return (x - ab.sqrt() * x_star) / (1 - ab).sqrt()

  for n in (100, 10):
    x0 = sch.sample(oracle, x_star.shape, "cpu", num_inference_steps=n)
    assert torch.allclose(x0, x_star, atol=1e-3), (n, (x0 - x_star).abs().max())
  noisy = sch.add_noise(x_star, torch.zeros_like(x_star), torch.zeros(3, dtype=torch.long))
  assert torch.allclose(noisy, x_star * sch.alphas_cumprod[0].float().sqrt())


def test_policy_learns_and_checkpoint_roundtrip():
  torch.manual_seed(0)
  obs = np.random.rand(20, 40, OBS_DIM).astype(np.float32)
  act = np.tanh(obs[..., :ACT_DIM] * 3 - 1.5).astype(np.float32)
  ow, aw = make_windows(obs, act, 2, 16, 8)
  on, an = MinMaxNormalizer.fit(obs), MinMaxNormalizer.fit(act)
  nobs, nact = torch.from_numpy(on.normalize(ow)), torch.from_numpy(an.normalize(aw))
  pol = DiffusionPolicy(OBS_DIM, ACT_DIM, **SMALL)
  pol.set_normalizers(on, an)
  ema = EMA(pol)
  opt = torch.optim.AdamW(pol.parameters(), lr=1e-3)
  losses = []
  for _ in range(100):
    idx = torch.randint(0, len(nobs), (32,))
    loss = pol.compute_loss(nobs[idx], nact[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    ema.update(pol)
    losses.append(loss.item())
  # Smoke-level check only (fixed seed, tiny net, 100 steps): the loss must go down.
  assert np.mean(losses[-20:]) < 0.95 * np.mean(losses[:20]), (losses[:5], losses[-5:])
  assert torch.equal(ema.model.act_scale, pol.act_scale)

  with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "ckpt.pt"
    save_checkpoint(path, ema.model, extra={"config": {"x": 1}})
    loaded, ckpt = load_checkpoint(path, "cpu")
    assert ckpt["config"] == {"x": 1}
    o = torch.from_numpy(obs[:3, :2])
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a1 = ema.model.predict_action(o, generator=g1)
    a2 = loaded.predict_action(o, generator=g2)
    assert a1.shape == (3, 16, ACT_DIM)
    assert torch.allclose(a1, a2, atol=1e-5)


def test_rollout_success_and_replay():
  env = FakeBatchedEnv()
  seeds = np.arange(6)
  r = rollout(env, seeds, GoToTarget(), T)
  assert r["obs"].shape == (6, T, OBS_DIM) and r["actions"].shape == (6, T, ACT_DIM)
  assert r["qpos"].shape == (6, T + 1, 16) and r["dist"].shape == (6, T + 1)
  assert r["phase"].shape == (6, T)
  assert r["success"].all()
  assert (r["first_success_step"] > 0).all()
  # obs[t] is the observation *before* action t
  np.testing.assert_allclose(r["obs"][:, 1, :3], r["box_pos"][:, 1], atol=1e-6)
  rr = rollout(env, seeds, ReplayController(r["actions"]), T)
  np.testing.assert_allclose(rr["obs"], r["obs"])
  # noise: executed actions differ from clean ones, and are clipped
  rn = rollout(env, seeds, GoToTarget(), T, noise_std=0.5, noise_rng=np.random.default_rng(0))
  assert np.abs(rn["actions"]).max() <= 1.0
  assert not np.allclose(rn["actions"], rn["actions_clean"])
  rows = episode_table(r, chunk=8)
  s = summarize(r, chunk=8)
  assert s["success_rate"] == 1.0 and rows[0]["failure_mode"] == "success"
  assert {"jerk_boundary", "jerk_within"} <= set(s)


def test_failure_labels():
  env = FakeBatchedEnv()
  idle = ReplayController(np.zeros((4, T, ACT_DIM), np.float32))
  r = rollout(env, np.arange(4), idle, T)
  s = summarize(r)
  assert s["success_rate"] == 0.0
  assert s["failure_modes"]["grasp_failed"] == 4   # gripper at box, box never lifted


def test_dp_controller_receding_horizon():
  env = FakeBatchedEnv()
  pol = DiffusionPolicy(OBS_DIM, ACT_DIM, **SMALL).eval()
  ctrl = DiffusionPolicyController(pol, "cpu", seed=0)
  r = rollout(env, np.arange(3), ctrl, 40)
  assert ctrl.num_inferences == 5  # ceil(40 / 8)
  assert np.abs(r["actions"]).max() <= 1.0


def test_config_overrides():
  cfg = utils.load_config("configs/dp_base.yaml", ["seed=3", "train.num_steps=10", "policy.down_dims=[8,16]"])
  assert cfg["seed"] == 3 and cfg["train"]["num_steps"] == 10 and cfg["policy"]["down_dims"] == [8, 16]
  sweep = utils.load_config("configs/sweep.yaml")
  assert [c["num_demos"] for c in sweep["conditions"]] == [100, 50, 25]
  lo, hi = utils.wilson_interval(50, 100)
  assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_train_resume_is_exact():
  """Interrupt-and-resume must produce exactly the same model as an uninterrupted run."""
  import train as train_mod
  with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    rng = np.random.default_rng(0)
    obs = rng.random((4, 30, OBS_DIM), dtype=np.float32)
    np.savez(d / "demos.npz", obs=obs, actions=np.tanh(obs[..., :ACT_DIM]), seeds=np.arange(4),
             success=np.ones(4, bool))
    cfg = utils.load_config("configs/dp_base.yaml", [
        f"data.path={(d / 'demos.npz').as_posix()}", "data.num_demos=4", "policy.down_dims=[8,16,32]",
        "policy.diffusion_step_embed_dim=8", "policy.num_train_timesteps=10", "train.num_steps=30",
        "train.batch_size=8", "train.warmup_steps=5", "train.log_every=5", "train.save_every=10"])
    train_mod.train(cfg, d / "full")
    assert train_mod.train(cfg, d / "resumed", stop_after=15) is None          # "disconnect" at step 15
    assert (d / "resumed" / "train_state.pt").exists()
    train_mod.train(cfg, d / "resumed")                                        # resume to the end
    assert not (d / "resumed" / "train_state.pt").exists()
    a = torch.load(d / "full" / "checkpoint.pt", weights_only=False)["state_dict"]
    b = torch.load(d / "resumed" / "checkpoint.pt", weights_only=False)["state_dict"]
    for k in a:
      assert torch.equal(a[k], b[k]), k
    loss_a = (d / "full" / "loss.csv").read_text().splitlines()
    loss_b = (d / "resumed" / "loss.csv").read_text().splitlines()
    assert [r.split(",")[:2] for r in loss_a] == [r.split(",")[:2] for r in loss_b]
    bad = dict(cfg, seed=1)
    (d / "resumed" / "checkpoint.pt").unlink()
    train_mod.train(bad, d / "resumed", stop_after=5)
    try:
      train_mod.train(cfg, d / "resumed")
      raise AssertionError("resuming with a different config must fail")
    except RuntimeError:
      pass


def test_analysis_and_demo_figures():
  import analyze
  import inspect_demos
  env = FakeBatchedEnv()
  r = rollout(env, np.arange(12), GoToTarget(), T)
  with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    # demo figures on fake data
    data = {k: v for k, v in r.items()}
    data["meta"] = np.array('{"name": "fake", "noise_std": 0.0, "teacher_success_rate": 1.0}')
    data["success"] = data["success"].copy()
    data["success"][0] = False
    stats = inspect_demos.compute_stats(data)
    assert stats["num_trajectories"] == 12 and stats["num_successful"] == 11
    from hw2 import plotting
    plotting.setup()
    for fn in ["fig_coverage", "fig_trajectories", "fig_actions", "fig_phases"]:
      assert getattr(inspect_demos, fn)(data, d / f"{fn}.png").exists()
    # sweep analysis on fake results
    analyze.RESULTS = d
    for name, rate in [("n100", 0.9), ("n50", 0.7), ("n25", 0.4)]:
      for seed in (0, 1):
        run = d / "runs" / f"{name}_s{seed}"
        s = summarize(r, chunk=8)
        s.update(success_rate=rate + 0.02 * seed, num_success=int(100 * rate), num_episodes=100)
        utils.save_json(s, run / "eval_summary.json")
        utils.save_json({"num_demos": 1}, run / "train_summary.json")
        (run / "loss.csv").write_text("step,loss,lr,elapsed_s\n100,0.5,1e-4,1\n200,0.2,1e-4,2\n")
    teacher = summarize(r, chunk=8)
    utils.save_json(teacher, d / "teacher_eval" / "eval_summary.json")
    utils.save_json({"teacher_success_rate": 0.95}, d / "demos" / "collection_baseline.json")
    sys.argv = ["analyze.py"]
    analyze.main()
    for f in ["summary_table.md", "fig_success_vs_demos.png", "fig_success_vs_threshold.png",
              "fig_loss_curves.png", "fig_failure_modes.png", "fig_smoothness.png"]:
      assert (d / f).exists(), f
    assert "| n25 | 25 |" in (d / "summary_table.md").read_text()


if __name__ == "__main__":
  tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
  import time
  for t in tests:
    t0 = time.time()
    t()
    print(f"ok {t.__name__} ({time.time() - t0:.1f}s)")
