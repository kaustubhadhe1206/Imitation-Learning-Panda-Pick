"""CSCI 645 HW2: imitation learning on MuJoCo Playground PandaPickCube.

Importing this package sets rendering / JAX memory environment variables, so
every entry-point script imports `hw2` before anything that pulls in mujoco or
jax.
"""

import os

# Headless OpenGL for mujoco.Renderer (Colab / CARC GPU nodes). Override with
# MUJOCO_GL=osmesa if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# JAX (simulator) and PyTorch (policy) share one GPU during evaluation, so JAX
# must not grab 75% of device memory up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
