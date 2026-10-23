"""Offline rendering of recorded trajectories (qpos + target) to frames / MP4.

Rendering from stored qpos (instead of from live MJX states) means any
demonstration or evaluation rollout can be re-rendered later from the saved
.npz without re-simulating. Uses the same default free camera as the course
README's GIFs.
"""

from pathlib import Path

import numpy as np


class TrajectoryRenderer:

  def __init__(self, mj_model, height=480, width=640, camera=-1):
    import mujoco
    self._mujoco = mujoco
    self.m = mj_model
    self.d = mujoco.MjData(mj_model)
    self.renderer = mujoco.Renderer(mj_model, height=height, width=width)
    self.camera = camera
    self.mocap = int(mj_model.body("mocap_target").mocapid[0])

  def render(self, qpos_seq, target_pos, every=1):
    frames = []
    for q in qpos_seq[::every]:
      self.d.qpos[:] = q
      self.d.qvel[:] = 0.0
      self.d.mocap_pos[self.mocap] = target_pos
      self.d.mocap_quat[self.mocap] = [1.0, 0.0, 0.0, 0.0]
      self._mujoco.mj_forward(self.m, self.d)
      self.renderer.update_scene(self.d, camera=self.camera)
      frames.append(self.renderer.render().copy())
    return frames

  def close(self):
    self.renderer.close()


def annotate(frames, text, color=(255, 255, 255)):
  """Burn a caption into the top-left corner of each frame (best effort)."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return frames
  out = []
  for f in frames:
    img = Image.fromarray(f)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 + 7 * len(text), 22], fill=(0, 0, 0))
    draw.text((4, 5), text, fill=color)
    out.append(np.asarray(img))
  return out


def tile(videos, cols):
  """Tile equal-length lists of frames into a grid video."""
  n = len(videos)
  rows = int(np.ceil(n / cols))
  length = max(len(v) for v in videos)
  h, w, _ = videos[0][0].shape
  blank = np.zeros((h, w, 3), dtype=np.uint8)
  out = []
  for t in range(length):
    grid = []
    for r in range(rows):
      row = []
      for c in range(cols):
        k = r * cols + c
        row.append(videos[k][min(t, len(videos[k]) - 1)] if k < n else blank)
      grid.append(np.concatenate(row, axis=1))
    out.append(np.concatenate(grid, axis=0))
  return out


def write_video(path, frames, fps):
  import imageio.v2 as imageio
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1)
  return path


def render_episode(renderer, result, i, fps_every=2, caption=None):
  """Frames for episode i of a rollout result (every `fps_every`-th state)."""
  frames = renderer.render(result["qpos"][i], result["target_pos"][i], every=fps_every)
  return annotate(frames, caption) if caption else frames
