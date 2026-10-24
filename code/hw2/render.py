"""Turning saved trajectories (joint positions + target) into images and MP4 videos.

We render from the saved qpos instead of from live simulator states, so any
demo or evaluation episode can be re-rendered later from the .npz file
without running the simulation again. The camera is the same default view as
in the course README GIFs.
"""

from pathlib import Path

import numpy as np


class TrajectoryRenderer:

  def __init__(self, mj_model, height=480, width=640, camera=-1):
    import mujoco
    self._mujoco = mujoco
    self.model = mj_model
    self.data = mujoco.MjData(mj_model)
    self.renderer = mujoco.Renderer(mj_model, height=height, width=width)
    self.camera = camera
    self.target_mocap_id = int(mj_model.body("mocap_target").mocapid[0])

  def render_frames(self, qpos_sequence, target_position, every_nth=1):
    """One image per saved state (every_nth=2 keeps every second state)."""
    frames = []
    for qpos in qpos_sequence[::every_nth]:
      self.data.qpos[:] = qpos
      self.data.qvel[:] = 0.0
      self.data.mocap_pos[self.target_mocap_id] = target_position
      self.data.mocap_quat[self.target_mocap_id] = [1.0, 0.0, 0.0, 0.0]
      self._mujoco.mj_forward(self.model, self.data)
      self.renderer.update_scene(self.data, camera=self.camera)
      frames.append(self.renderer.render().copy())
    return frames

  def close(self):
    self.renderer.close()


def add_caption(frames, text, color=(255, 255, 255)):
  """Write a caption in the top-left corner of every frame (skipped if Pillow is missing)."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return frames
  captioned = []
  for frame in frames:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 8 + 7 * len(text), 22], fill=(0, 0, 0))
    draw.text((4, 5), text, fill=color)
    captioned.append(np.asarray(image))
  return captioned


def tile_videos(videos, num_columns):
  """Put several videos (lists of frames) next to each other in a grid."""
  num_videos = len(videos)
  num_rows = int(np.ceil(num_videos / num_columns))
  num_frames = max(len(video) for video in videos)
  height, width, _ = videos[0][0].shape
  blank = np.zeros((height, width, 3), dtype=np.uint8)
  tiled = []
  for t in range(num_frames):
    rows = []
    for row in range(num_rows):
      tiles = []
      for col in range(num_columns):
        k = row * num_columns + col
        tiles.append(videos[k][min(t, len(videos[k]) - 1)] if k < num_videos else blank)
      rows.append(np.concatenate(tiles, axis=1))
    tiled.append(np.concatenate(rows, axis=0))
  return tiled


def save_video(path, frames, fps):
  import imageio.v2 as imageio
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1)
  return path
