"""CSCI 645 HW2: imitation learning on MuJoCo Playground PandaPickCube.

Importing this package sets a few environment variables, so every script
imports `hw2` before anything that loads mujoco or jax.
"""

import os

# Render without a screen (Colab / CARC GPU nodes). Use MUJOCO_GL=osmesa if EGL doesn't work.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# The simulator (JAX) and the policy (PyTorch) share one GPU during evaluation,
# so stop JAX from reserving 75% of the GPU memory at startup.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
