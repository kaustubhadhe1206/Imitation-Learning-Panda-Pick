"""PandaPickCube from MuJoCo Playground, run as a batch of environments in parallel.

We use the environment exactly as it comes (JAX backend, like in the course
README). Every episode starts from env.reset(jax.random.PRNGKey(seed)), so any
demo or evaluation episode can be reproduced from its seed number alone.

JAX is only imported inside the functions below. That way the other modules
(rollout, dataset, policy) can be tested on a laptop without JAX installed.
"""

import numpy as np

ENV_NAME = "PandaPickCube"


def load_pick_cube_env(impl="jax"):
  from mujoco_playground import registry
  return registry.load(ENV_NAME, config_overrides={"impl": impl})


def get_env_config():
  """Default PandaPickCube settings (control step, episode length, action scale, ...) as a dict."""
  from mujoco_playground import registry
  return registry.get_default_config(ENV_NAME).to_dict()


class BatchedPickCubeEnv:
  """Runs many PandaPickCube episodes at once on the GPU (jax.vmap + jax.jit).

  reset(seeds) and step(state, actions) work on the whole batch.
  to_numpy(state) copies what the controllers need to the CPU as numpy arrays.
  """

  def __init__(self, env=None, impl="jax"):
    import jax
    import jax.numpy as jnp

    self._jax = jax
    self._jnp = jnp
    self.env = env if env is not None else load_pick_cube_env(impl)
    self.mj_model = self.env.mj_model
    env_config = get_env_config()
    self.episode_length = int(env_config["episode_length"])   # 150 steps
    self.action_scale = float(env_config["action_scale"])     # ctrl += 0.04 * action
    self.control_dt = float(env_config["ctrl_dt"])            # 0.02 s per step
    self.action_size = int(self.env.action_size)              # 8 = 7 arm joints + gripper

    gripper_site_id = self.mj_model.site("gripper").id
    box_body_id = self.mj_model.body("box").id

    def pick_useful_fields(state):
      return {
          "obs": state.obs,
          "qpos": state.data.qpos,
          "ctrl": state.data.ctrl,
          "grip_pos": state.data.site_xpos[:, gripper_site_id],
          "box_pos": state.data.xpos[:, box_body_id],
          "target_pos": state.info["target_pos"],
          "reward": state.reward,
          "done": state.done,
          "reached_box": state.info["reached_box"],
      }

    self._reset_batch = jax.jit(jax.vmap(self.env.reset))
    self._step_batch = jax.jit(jax.vmap(self.env.step))
    self._pick_useful_fields = jax.jit(pick_useful_fields)

  def reset(self, seeds):
    jax, jnp = self._jax, self._jnp
    seeds = jnp.asarray(np.asarray(seeds), dtype=jnp.int32)
    random_keys = jax.vmap(jax.random.PRNGKey)(seeds)
    return self._reset_batch(random_keys)

  def step(self, state, actions):
    return self._step_batch(state, self._jnp.asarray(actions, dtype=self._jnp.float32))

  def to_numpy(self, state):
    fields = self._jax.device_get(self._pick_useful_fields(state))
    return {name: np.asarray(value) for name, value in fields.items()}
