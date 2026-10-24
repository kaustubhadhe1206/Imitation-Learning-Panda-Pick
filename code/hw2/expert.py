"""Scripted pick-and-place expert (the "teacher") for PandaPickCube.

How actions work in PandaPickCube: the action is 8 numbers in [-1, 1] and the
env does ctrl = clip(ctrl + 0.04 * action). The first 7 ctrl values are target
angles for the 7 arm joints (position servos), the 8th is the finger opening
(0 = closed, 0.04 m = open).

The expert goes through these phases:
  APPROACH   move above the box with the gripper open
  DESCEND    go straight down until the gripper is around the box
  GRASP      stay still and close the gripper
  LIFT       lift the box straight up a bit
  TRANSPORT  move so the box center is on the target, then hold it there

Each phase has a waypoint for the gripper. Instead of jumping to the waypoint,
a "reference point" moves toward it with a speed limit, so the motion is
smooth. Inverse kinematics (IK) turns the reference point into joint angles,
keeping the gripper pointing straight down with the fingers lined up with the
box sides.

One thing I had to handle: the Panda model in this env has no gravity
compensation, so if we just command the joint angles we want, the arm sags a
few cm under its own weight and misses the 4 cm box. So we compute the gravity
torque at the desired angles and command angle + torque / kp instead, which is
the target where the servo exactly cancels gravity.

If the box slips out of the gripper while lifting/moving, the expert goes back
to APPROACH and tries again.

All of this runs on the CPU with plain MuJoCo, using the same model as the env.
"""

import mujoco
import numpy as np

APPROACH, DESCEND, GRASP, LIFT, TRANSPORT = range(5)
PHASE_NAMES = ["approach", "descend", "grasp", "lift", "transport"]

DEFAULT_EXPERT_CONFIG = {
    "pregrasp_height": 0.06,          # m above the box center to stop before going down
    "grasp_height_offset": 0.0,       # where the gripper point goes relative to the box center
    "lift_height": 0.06,              # m to lift straight up after grasping
    "approach_speed": 0.02,           # max m per control step (one step = 0.02 s)
    "descend_speed": 0.01,
    "lift_speed": 0.012,
    "transport_speed": 0.015,
    "reference_gain": 0.35,           # reference point covers this fraction of the remaining distance per step
    "reference_tolerance": 0.003,     # reference point counts as "at the waypoint" within this distance
    "approach_tolerance": 0.03,       # real gripper must be this close to the pregrasp point to start descending
    "grasp_xy_tolerance": 0.008,      # how well centered over the box before closing
    "grasp_z_tolerance": 0.01,
    "descend_timeout_steps": 30,      # close the gripper anyway after this many steps of descending
    "gripper_close_steps": 8,         # steps to wait while closing before lifting
    "gripper_max_action": 0.5,        # limit on the gripper action so it closes over ~2 steps
    "drop_distance": 0.03,            # if the box moves this far from where it was grasped, regrasp
    "ik_iterations": 2,
    "ik_damping": 0.05,
    "orientation_gain": 1.0,
    "nullspace_gain": 0.05,           # gentle pull toward the home pose using the 7th (extra) joint
    "max_joint_step": 0.1,            # rad per IK iteration
    "use_gravity_compensation": True,
}


def orientation_error(current_rotation, target_rotation):
  """Small rotation (axis * angle, world frame) that turns current_rotation toward target_rotation."""
  error = np.zeros(3)
  for axis in range(3):
    error += np.cross(current_rotation[:, axis], target_rotation[:, axis])
  return 0.5 * error


def make_downward_rotation(home_rotation):
  """Gripper pointing straight down, turned (in 90 degree steps) so the fingers close along a world axis."""
  x_axis = home_rotation[:, 0].copy()
  x_axis[2] = 0.0
  yaw = np.arctan2(x_axis[1], x_axis[0])
  yaw = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
  x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
  z_axis = np.array([0.0, 0.0, -1.0])
  y_axis = np.cross(z_axis, x_axis)
  return np.column_stack([x_axis, y_axis, z_axis])


class ScriptedPickPlaceExpert:
  """The expert for ONE environment (BatchedExpertController below runs one per episode)."""

  def __init__(self, mj_model, action_scale, config=None):
    self.config = {**DEFAULT_EXPERT_CONFIG, **(config or {})}
    self.model = mj_model
    self.data = mujoco.MjData(mj_model)
    self.action_scale = action_scale

    arm_joint_ids = [mj_model.joint(f"joint{i}").id for i in range(1, 8)]
    self.arm_qpos_index = mj_model.jnt_qposadr[arm_joint_ids]
    self.arm_dof_index = mj_model.jnt_dofadr[arm_joint_ids]
    self.joint_min, self.joint_max = mj_model.jnt_range[arm_joint_ids].T.copy()
    self.ctrl_min, self.ctrl_max = mj_model.actuator_ctrlrange.T.copy()
    self.servo_kp = mj_model.actuator_gainprm[:7, 0].copy()   # position servo stiffness per joint
    self.gripper_site_id = mj_model.site("gripper").id

    home_keyframe = mj_model.keyframe("home")
    self.data.qpos[:] = home_keyframe.qpos
    self.home_joint_angles = home_keyframe.qpos[self.arm_qpos_index].copy()
    mujoco.mj_kinematics(mj_model, self.data)
    home_rotation = self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
    self.target_rotation = make_downward_rotation(home_rotation)

    self._position_jacobian = np.zeros((3, mj_model.nv))
    self._rotation_jacobian = np.zeros((3, mj_model.nv))
    self._bias_forces = np.zeros(mj_model.nv)
    self.phase = APPROACH

  # --- kinematics and gravity ----------------------------------------------

  def forward_kinematics(self, joint_angles):
    """Gripper position and rotation matrix for the given arm joint angles."""
    self.data.qpos[self.arm_qpos_index] = joint_angles
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)
    position = self.data.site_xpos[self.gripper_site_id].copy()
    rotation = self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
    return position, rotation

  def solve_ik(self, joint_angles, target_position):
    """A few damped least-squares IK steps toward target_position with the gripper pointing down."""
    cfg = self.config
    identity_6, identity_7 = np.eye(6), np.eye(7)
    for _ in range(cfg["ik_iterations"]):
      position, rotation = self.forward_kinematics(joint_angles)
      error = np.concatenate([target_position - position,
                              cfg["orientation_gain"] * orientation_error(rotation, self.target_rotation)])
      mujoco.mj_jacSite(self.model, self.data, self._position_jacobian, self._rotation_jacobian,
                        self.gripper_site_id)
      jacobian = np.vstack([self._position_jacobian[:, self.arm_dof_index],
                            self._rotation_jacobian[:, self.arm_dof_index]])
      damping = cfg["ik_damping"] ** 2 * identity_6
      jacobian_pinv = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, identity_6)
      joint_step = jacobian_pinv @ error
      # Use the leftover (7th) degree of freedom to stay close to the home pose.
      nullspace = identity_7 - jacobian_pinv @ jacobian
      joint_step += nullspace @ (cfg["nullspace_gain"] * (self.home_joint_angles - joint_angles))
      joint_step = np.clip(joint_step, -cfg["max_joint_step"], cfg["max_joint_step"])
      joint_angles = np.clip(joint_angles + joint_step, self.joint_min + 1e-3, self.joint_max - 1e-3)
    return joint_angles

  def gravity_torques(self, joint_angles):
    """Torque each arm joint needs to hold the arm still at these angles."""
    self.data.qpos[self.arm_qpos_index] = joint_angles
    self.data.qvel[:] = 0.0
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)
    mujoco.mj_comVel(self.model, self.data)
    mujoco.mj_rne(self.model, self.data, 0, self._bias_forces)   # zero velocity -> only gravity
    return self._bias_forces[self.arm_dof_index].copy()

  # --- control ---------------------------------------------------------------

  def set_phase(self, phase):
    self.phase = phase
    self.steps_in_phase = 0

  def reference_reached(self, waypoint):
    return np.linalg.norm(self.reference_point - waypoint) < self.config["reference_tolerance"]

  def reset(self, env_state):
    self.set_phase(APPROACH)
    self.reference_joint_angles = env_state["qpos"][self.arm_qpos_index].astype(np.float64).copy()
    self.reference_point = env_state["grip_pos"].astype(np.float64).copy()
    self.grasp_waypoint = None
    self.lift_waypoint = None
    self.grasp_offset = np.zeros(3)     # gripper position minus box position at the moment of grasping

  def get_action(self, env_state):
    cfg = self.config
    gripper_pos = env_state["grip_pos"].astype(np.float64)
    box_pos = env_state["box_pos"].astype(np.float64)
    target_pos = env_state["target_pos"].astype(np.float64)
    self.data.qpos[:] = env_state["qpos"]
    up = np.array([0.0, 0.0, 1.0])

    # 1) Decide if we should move on to the next phase (using the current measurements).
    if self.phase == APPROACH:
      pregrasp_point = box_pos + cfg["pregrasp_height"] * up
      if (self.reference_reached(pregrasp_point)
          and np.linalg.norm(gripper_pos - pregrasp_point) < cfg["approach_tolerance"]):
        self.set_phase(DESCEND)
    elif self.phase == DESCEND:
      grasp_point = box_pos + cfg["grasp_height_offset"] * up
      centered = (np.linalg.norm(gripper_pos[:2] - grasp_point[:2]) < cfg["grasp_xy_tolerance"]
                  and abs(gripper_pos[2] - grasp_point[2]) < cfg["grasp_z_tolerance"])
      timed_out = self.steps_in_phase >= cfg["descend_timeout_steps"]
      if (self.reference_reached(grasp_point) and centered) or timed_out:
        self.grasp_waypoint = self.reference_point.copy()
        self.set_phase(GRASP)
    elif self.phase == GRASP:
      if self.steps_in_phase >= cfg["gripper_close_steps"]:
        self.grasp_offset = gripper_pos - box_pos
        self.lift_waypoint = self.reference_point + cfg["lift_height"] * up
        self.set_phase(LIFT)
    elif self.phase == LIFT:
      if self.reference_reached(self.lift_waypoint):
        self.set_phase(TRANSPORT)
    if self.phase in (LIFT, TRANSPORT):
      box_slipped = np.linalg.norm(gripper_pos - self.grasp_offset - box_pos) > cfg["drop_distance"]
      if box_slipped:
        self.set_phase(APPROACH)   # go back and grasp again

    # 2) Waypoint, speed limit and gripper command for the current phase.
    if self.phase == APPROACH:
      waypoint, speed, close_gripper = box_pos + cfg["pregrasp_height"] * up, cfg["approach_speed"], False
    elif self.phase == DESCEND:
      waypoint, speed, close_gripper = box_pos + cfg["grasp_height_offset"] * up, cfg["descend_speed"], False
    elif self.phase == GRASP:
      waypoint, speed, close_gripper = self.grasp_waypoint, cfg["descend_speed"], True
    elif self.phase == LIFT:
      waypoint, speed, close_gripper = self.lift_waypoint, cfg["lift_speed"], True
    else:
      waypoint, speed, close_gripper = target_pos + self.grasp_offset, cfg["transport_speed"], True

    # 3) Move the reference point toward the waypoint (slowing down near it).
    to_waypoint = waypoint - self.reference_point
    distance = np.linalg.norm(to_waypoint)
    if distance > 1e-9:
      step_length = min(speed, cfg["reference_gain"] * distance)
      self.reference_point = self.reference_point + to_waypoint / distance * step_length

    # 4) Joint targets from IK, plus the gravity correction.
    self.reference_joint_angles = self.solve_ik(self.reference_joint_angles, self.reference_point)
    desired_ctrl = self.reference_joint_angles.copy()
    if cfg["use_gravity_compensation"]:
      desired_ctrl += self.gravity_torques(self.reference_joint_angles) / self.servo_kp
    desired_ctrl = np.clip(desired_ctrl, self.ctrl_min[:7], self.ctrl_max[:7])

    # 5) Turn "where ctrl should be" into an action (= change of ctrl / 0.04, clipped to [-1, 1]).
    action = np.zeros(8, dtype=np.float32)
    action[:7] = np.clip((desired_ctrl - env_state["ctrl"][:7]) / self.action_scale, -1.0, 1.0)
    finger_target = self.ctrl_min[7] if close_gripper else self.ctrl_max[7]
    action[7] = np.clip((finger_target - env_state["ctrl"][7]) / self.action_scale,
                        -cfg["gripper_max_action"], cfg["gripper_max_action"])
    self.steps_in_phase += 1
    return action


def get_single_env_state(batched_state, index):
  return {name: values[index] for name, values in batched_state.items()}


class BatchedExpertController:
  """Runs one ScriptedPickPlaceExpert (each with its own MjData) per episode in the batch."""

  def __init__(self, mj_model, action_scale, config=None):
    self.mj_model = mj_model
    self.action_scale = action_scale
    self.config = config
    self.experts = []
    self.current_phases = np.zeros(0, dtype=np.int8)

  def reset(self, env_state):
    batch_size = len(env_state["obs"])
    while len(self.experts) < batch_size:
      self.experts.append(ScriptedPickPlaceExpert(self.mj_model, self.action_scale, self.config))
    self.experts = self.experts[:batch_size]
    for i, expert in enumerate(self.experts):
      expert.reset(get_single_env_state(env_state, i))
    self.current_phases = np.array([expert.phase for expert in self.experts], dtype=np.int8)

  def get_action(self, env_state, step):
    actions = np.stack([expert.get_action(get_single_env_state(env_state, i))
                        for i, expert in enumerate(self.experts)])
    self.current_phases = np.array([expert.phase for expert in self.experts], dtype=np.int8)
    return actions
