"""Scripted pick-and-place expert for PandaPickCube.

Action interface (MuJoCo Playground PandaPickCube): an 8-D action in [-1, 1]
changes the actuator position targets, ctrl <- clip(ctrl + 0.04 * a). The
first 7 actuators are joint position servos (kp = 1000/1000/750/750/300/300/300),
the 8th drives the finger opening in [0, 0.04] m.

The expert is a phase state machine over a Cartesian waypoint for the gripper
site (the point between the fingertips):

  APPROACH   above the box (pregrasp_height), gripper open
  DESCEND    down to the box centre, gripper open
  GRASP      hold position while closing the gripper
  LIFT       straight up by lift_height
  TRANSPORT  move so that the box centre sits on the target, hold there

A reference point p_ref moves toward the waypoint with a per-phase speed cap.
A reference joint configuration q_ref tracks p_ref with damped-least-squares
IK (position + a fixed downward orientation with the fingers aligned to the
box faces, null-space pull to the home posture). The arm has no gravity
compensation in this model, so a position servo sags under load; we therefore
command ctrl = q_ref + g(q_ref) / kp, the static target at which the servo
torque cancels gravity, and the action is the (rate-limited) change from the
current ctrl to that target. If the box slips out of the grasp during LIFT or
TRANSPORT the expert re-approaches it.

Everything here runs on CPU with the same mj_model the MJX env was built from.
"""

import mujoco
import numpy as np

APPROACH, DESCEND, GRASP, LIFT, TRANSPORT = range(5)
PHASE_NAMES = ["approach", "descend", "grasp", "lift", "transport"]

DEFAULT_EXPERT_CFG = {
    "pregrasp_height": 0.06,   # m above box centre
    "grasp_dz": 0.0,           # gripper site relative to box centre when grasping
    "lift_height": 0.06,       # m
    "v_approach": 0.02,        # m per control step (0.02 s)
    "v_descend": 0.01,
    "v_lift": 0.012,
    "v_transport": 0.015,
    "ref_gain": 0.35,          # fraction of remaining distance p_ref covers per step
    "ref_tol": 0.003,          # p_ref considered at the waypoint
    "approach_tol": 0.03,      # actual gripper within this of the pregrasp point
    "grasp_xy_tol": 0.008,
    "grasp_z_tol": 0.01,
    "descend_timeout": 30,     # steps
    "close_steps": 8,          # steps spent closing before lifting
    "gripper_rate": 0.5,       # |action| cap for the finger actuator
    "drop_tol": 0.03,          # box-vs-grasp offset that triggers a regrasp
    "ik_iters": 2,
    "ik_damping": 0.05,
    "rot_gain": 1.0,
    "null_gain": 0.05,
    "max_dq": 0.1,             # rad per IK iteration
    "gravity_comp": True,
}


def _rot_err(R, R_des):
  """Small-angle orientation error (world frame) rotating R toward R_des."""
  return 0.5 * (np.cross(R[:, 0], R_des[:, 0]) + np.cross(R[:, 1], R_des[:, 1])
                + np.cross(R[:, 2], R_des[:, 2]))


def _downward_rotation(R_home):
  """Gripper pointing straight down, yaw snapped so fingers close along a world axis."""
  x = R_home[:, 0].copy()
  x[2] = 0.0
  yaw = np.arctan2(x[1], x[0])
  yaw = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
  x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
  z = np.array([0.0, 0.0, -1.0])
  y = np.cross(z, x)
  return np.column_stack([x, y, z])


class ScriptedExpert:
  """Expert for a single environment instance."""

  def __init__(self, mj_model, action_scale, cfg=None):
    self.cfg = {**DEFAULT_EXPERT_CFG, **(cfg or {})}
    m = self.m = mj_model
    self.d = mujoco.MjData(m)
    self.action_scale = action_scale

    jids = [m.joint(f"joint{i}").id for i in range(1, 8)]
    self.qadr = m.jnt_qposadr[jids]
    self.dadr = m.jnt_dofadr[jids]
    self.jlo, self.jhi = m.jnt_range[jids].T.copy()
    self.clo, self.chi = m.actuator_ctrlrange.T.copy()
    self.kp = m.actuator_gainprm[:7, 0].copy()
    self.site = m.site("gripper").id

    key = m.keyframe("home")
    self.d.qpos[:] = key.qpos
    self.q_home = key.qpos[self.qadr].copy()
    mujoco.mj_kinematics(m, self.d)
    self.R_des = _downward_rotation(self.d.site_xmat[self.site].reshape(3, 3).copy())

    self._jacp = np.zeros((3, m.nv))
    self._jacr = np.zeros((3, m.nv))
    self._bias = np.zeros(m.nv)
    self.phase = APPROACH

  # -- kinematics / dynamics helpers ---------------------------------------

  def _fk(self, q):
    self.d.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)
    return (self.d.site_xpos[self.site].copy(),
            self.d.site_xmat[self.site].reshape(3, 3).copy())

  def _ik(self, q, p_target):
    c = self.cfg
    eye6, eye7 = np.eye(6), np.eye(7)
    for _ in range(c["ik_iters"]):
      p, R = self._fk(q)
      err = np.concatenate([p_target - p, c["rot_gain"] * _rot_err(R, self.R_des)])
      mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.site)
      J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
      J_pinv = J.T @ np.linalg.solve(J @ J.T + c["ik_damping"] ** 2 * eye6, eye6)
      dq = J_pinv @ err + (eye7 - J_pinv @ J) @ (c["null_gain"] * (self.q_home - q))
      dq = np.clip(dq, -c["max_dq"], c["max_dq"])
      q = np.clip(q + dq, self.jlo + 1e-3, self.jhi - 1e-3)
    return q

  def _gravity(self, q):
    self.d.qpos[self.qadr] = q
    self.d.qvel[:] = 0.0
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)
    mujoco.mj_comVel(self.m, self.d)
    mujoco.mj_rne(self.m, self.d, 0, self._bias)
    return self._bias[self.dadr].copy()

  # -- control ---------------------------------------------------------------

  def _set_phase(self, phase):
    self.phase = phase
    self.phase_t = 0

  def _ref_reached(self, w):
    return np.linalg.norm(self.p_ref - w) < self.cfg["ref_tol"]

  def reset(self, s):
    self._set_phase(APPROACH)
    self.q_ref = s["qpos"][self.qadr].astype(np.float64).copy()
    self.p_ref = s["grip_pos"].astype(np.float64).copy()
    self.grasp_w = None
    self.lift_w = None
    self.offset = np.zeros(3)

  def act(self, s):
    c = self.cfg
    g = s["grip_pos"].astype(np.float64)
    b = s["box_pos"].astype(np.float64)
    tgt = s["target_pos"].astype(np.float64)
    self.d.qpos[:] = s["qpos"]
    up = np.array([0.0, 0.0, 1.0])

    # Phase transitions (based on the current measurement).
    if self.phase == APPROACH:
      w = b + c["pregrasp_height"] * up
      if self._ref_reached(w) and np.linalg.norm(g - w) < c["approach_tol"]:
        self._set_phase(DESCEND)
    elif self.phase == DESCEND:
      w = b + c["grasp_dz"] * up
      aligned = (np.linalg.norm(g[:2] - w[:2]) < c["grasp_xy_tol"]
                 and abs(g[2] - w[2]) < c["grasp_z_tol"])
      if (self._ref_reached(w) and aligned) or self.phase_t >= c["descend_timeout"]:
        self.grasp_w = self.p_ref.copy()
        self._set_phase(GRASP)
    elif self.phase == GRASP:
      if self.phase_t >= c["close_steps"]:
        self.offset = g - b
        self.lift_w = self.p_ref + c["lift_height"] * up
        self._set_phase(LIFT)
    elif self.phase == LIFT:
      if self._ref_reached(self.lift_w):
        self._set_phase(TRANSPORT)
    if self.phase in (LIFT, TRANSPORT) and np.linalg.norm(g - self.offset - b) > c["drop_tol"]:
      self._set_phase(APPROACH)  # box slipped: go back and regrasp

    # Waypoint, speed cap and gripper command for the (possibly new) phase.
    if self.phase == APPROACH:
      w, v, close = b + c["pregrasp_height"] * up, c["v_approach"], False
    elif self.phase == DESCEND:
      w, v, close = b + c["grasp_dz"] * up, c["v_descend"], False
    elif self.phase == GRASP:
      w, v, close = self.grasp_w, c["v_descend"], True
    elif self.phase == LIFT:
      w, v, close = self.lift_w, c["v_lift"], True
    else:
      w, v, close = tgt + self.offset, c["v_transport"], True

    delta = w - self.p_ref
    dist = np.linalg.norm(delta)
    if dist > 1e-9:
      self.p_ref = self.p_ref + delta / dist * min(v, c["ref_gain"] * dist)

    self.q_ref = self._ik(self.q_ref, self.p_ref)
    ctrl_cmd = self.q_ref.copy()
    if c["gravity_comp"]:
      ctrl_cmd += self._gravity(self.q_ref) / self.kp
    ctrl_cmd = np.clip(ctrl_cmd, self.clo[:7], self.chi[:7])

    a = np.zeros(8, dtype=np.float32)
    a[:7] = np.clip((ctrl_cmd - s["ctrl"][:7]) / self.action_scale, -1.0, 1.0)
    finger_target = self.clo[7] if close else self.chi[7]
    a[7] = np.clip((finger_target - s["ctrl"][7]) / self.action_scale,
                   -c["gripper_rate"], c["gripper_rate"])
    self.phase_t += 1
    return a


def _index(snap, i):
  return {k: v[i] for k, v in snap.items()}


class ExpertController:
  """Batched wrapper: one ScriptedExpert (with its own MjData) per environment."""

  def __init__(self, mj_model, action_scale, cfg=None):
    self.mj_model = mj_model
    self.action_scale = action_scale
    self.cfg = cfg
    self.experts = []
    self.phase = np.zeros(0, dtype=np.int8)

  def reset(self, snap):
    n = len(snap["obs"])
    while len(self.experts) < n:
      self.experts.append(ScriptedExpert(self.mj_model, self.action_scale, self.cfg))
    self.experts = self.experts[:n]
    for i, e in enumerate(self.experts):
      e.reset(_index(snap, i))
    self.phase = np.array([e.phase for e in self.experts], dtype=np.int8)

  def act(self, snap, t):
    actions = np.stack([e.act(_index(snap, i)) for i, e in enumerate(self.experts)])
    self.phase = np.array([e.phase for e in self.experts], dtype=np.int8)
    return actions
