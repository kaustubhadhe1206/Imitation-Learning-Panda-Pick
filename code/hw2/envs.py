"""Batched MuJoCo Playground PandaPickCube (JAX/MJX backend).

The environment is used unmodified: `registry.load("PandaPickCube")` with the
JAX implementation, exactly as in the course README. Each environment instance
is reset from an integer seed via `jax.random.PRNGKey(seed)`, so any episode
(demonstration or evaluation) can be reproduced from its seed alone.

JAX is imported lazily so that the numpy-only modules (rollout, dataset,
policy) can be unit-tested on machines without JAX.
"""

import numpy as np

ENV_NAME = "PandaPickCube"


def load_env(impl="jax"):
  from mujoco_playground import registry
  return registry.load(ENV_NAME, config_overrides={"impl": impl})


def env_config_dict():
  from mujoco_playground import registry
  return registry.get_default_config(ENV_NAME).to_dict()


class BatchedEnv:
  """jit(vmap(...)) reset/step over a batch of seeds + numpy snapshots."""

  def __init__(self, env=None, impl="jax"):
    import jax
    import jax.numpy as jnp

    self._jax, self._jnp = jax, jnp
    self.env = env if env is not None else load_env(impl)
    self.mj_model = self.env.mj_model
    cfg = env_config_dict()
    self.episode_length = int(cfg["episode_length"])
    self.action_scale = float(cfg["action_scale"])
    self.ctrl_dt = float(cfg["ctrl_dt"])
    self.action_size = int(self.env.action_size)

    gsite = self.mj_model.site("gripper").id
    box = self.mj_model.body("box").id

    def extract(s):
      return {
          "obs": s.obs,
          "qpos": s.data.qpos,
          "ctrl": s.data.ctrl,
          "grip_pos": s.data.site_xpos[:, gsite],
          "box_pos": s.data.xpos[:, box],
          "target_pos": s.info["target_pos"],
          "reward": s.reward,
          "done": s.done,
          "reached_box": s.info["reached_box"],
      }

    self._reset = jax.jit(jax.vmap(self.env.reset))
    self._step = jax.jit(jax.vmap(self.env.step))
    self._extract = jax.jit(extract)

  def reset(self, seeds):
    jax, jnp = self._jax, self._jnp
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(np.asarray(seeds), dtype=jnp.int32))
    return self._reset(keys)

  def step(self, state, actions):
    return self._step(state, self._jnp.asarray(actions, dtype=self._jnp.float32))

  def snapshot(self, state):
    out = self._jax.device_get(self._extract(state))
    return {k: np.asarray(v) for k, v in out.items()}
