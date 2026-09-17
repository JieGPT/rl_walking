import os

import mujoco
import numpy as np
import transforms3d as tf3

from tasks import cmd_walking_task

from .g1_base import G1BaseEnv
from .gen_xml import ARM_JOINTS, WAIST_JOINTS, builder


class G1CmdWalkEnv(G1BaseEnv):
  def _get_default_config_path(self):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs/cmd_walk.yaml")

  def _build_xml(self) -> str:
    export_dir = self._get_xml_export_dir("g1_cmd_walk")
    path_to_xml = os.path.join(export_dir, "g1.xml")
    if not os.path.exists(path_to_xml):
      builder(
        export_dir,
        config={
          "unused_joints": [WAIST_JOINTS, ARM_JOINTS],
          "ctrllimited": self.cfg.ctrllimited,
          "jointlimited": self.cfg.jointlimited,
          "minimal": self.cfg.reduced_xml,
        },
      )
    return path_to_xml

  def _setup_task(self, control_dt: float) -> None:
    task_cfg = self.cfg.task
    self.task = cmd_walking_task.CmdWalkingTask(
      client=self.interface,
      dt=control_dt,
      neutral_pose=self.half_sitting_pose,
      root_body="pelvis",
      lfoot_body=self.LFOOT_BODY,
      rfoot_body=self.RFOOT_BODY,
      head_body="torso_link",
    )
    self.task._goal_height_ref = task_cfg.goal_height
    self.task._total_duration = task_cfg.total_duration
    self.task._swing_duration = task_cfg.swing_duration
    self.task._stance_duration = task_cfg.stance_duration

  def _setup_robot(self) -> None:
    super()._setup_robot()
    self._setup_mirror_indices()

  def _setup_mirror_indices(self) -> None:
        # Mirror indices over the 41-D robot state + 9-D external state.
        # Order: root_orient(2), root_ang_vel(3), motor_pos(12), motor_vel(12),
        # motor_tau(12), clock(2), mode_one_hot(4), mode_ref(3).
        # Joint order in motor blocks: left(6) then right(6); within a leg
        # (model actuator order): hip_pitch, hip_roll, hip_yaw, knee,
        # ankle_pitch, ankle_roll.
        # Sign flips: hip_roll, hip_yaw and ankle_roll (rotations about the
        # body's x/y axes).
        base_mir_obs = [
            -0.1,
            1,  # root orient (roll, pitch)
            -2,
            3,
            -4,  # root ang vel
            11,
            -12,
            -13,
            14,
            15,
            -16,  # motor pos [1] -> right leg
            5,
            -6,
            -7,
            8,
            9,
            -10,  # motor pos [2] -> left leg
            23,
            -24,
            -25,
            26,
            27,
            -28,  # motor vel [1]
            17,
            -18,
            -19,
            20,
            21,
            -22,  # motor vel [2]
            35,
            -36,
            -37,
            38,
            39,
            -40,  # motor torque [1]
            29,
            -30,
            -31,
            32,
            33,
            -34,  # motor torque [2]
        ]
        # External state layout: clock(2), mode_one_hot(4), mode_ref(3).
        # Left-right mirroring swaps the turn modes and flips the yaw
        # command sign; the gait clock and forward velocity are invariant.
        n_robot = len(base_mir_obs)
        clock = [n_robot, n_robot + 1]
        one_hot = [n_robot + 2, n_robot + 3, n_robot + 5, n_robot + 4]  # TURN_LEFT <-> TURN_RIGHT
        mode_ref = [-(n_robot + 6), n_robot + 7, n_robot + 8]  # yaw sign flip
        append_obs = clock + one_hot + mode_ref
        self.robot.clock_inds = clock
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        # Action ordering: [left_leg(6), right_leg(6)]; mirror swaps legs and
        # flips signs of roll/yaw/ankle_roll dofs.
        self.robot.mirrored_acts = [6, -7, -8, 9, 10, -11, 0.1, -1, -2, 3, 4, -5]

  def _get_num_external_obs(self):
    return 9  # clock(2) + one-hot(4) + mode_ref(3)

  def _get_external_state(self) -> np.ndarray:
    clock = [
      np.sin(2 * np.pi * self.task._phase / self.task._period),
      np.cos(2 * np.pi * self.task._phase / self.task._period),
    ]
    return np.concatenate((clock, self.task.mode.encode(), self.task.mode_ref))

  def _setup_obs_normalization(self) -> None:
    self.obs_mean = np.concatenate(
      (
        np.zeros(5),              # root roll/pitch + ang vel
        self.half_sitting_pose,   # joint pos (12)
        np.zeros(12),             # joint vel
        np.zeros(12),             # joint torque
        [0, 0],                   # gait clock
        [0.25, 0.25, 0.25, 0.25], # mode one-hot (4)
        [0, 0, 0],                # mode_ref [yaw, vx, vy]
      )
    )
    self.obs_std = np.concatenate(
      (
        [0.2, 0.2, 1, 1, 1],
        0.5 * np.ones(12),
        4 * np.ones(12),
        100 * np.ones(12),
        [1, 1],
        [1, 1, 1, 1],
        [0.5, 0.5, 0.5],
      )
    )
    self.obs_mean = np.tile(self.obs_mean, self.history_len)
    self.obs_std = np.tile(self.obs_std, self.history_len)

  def draw_markers(self, marker_drawer):
    if not hasattr(self.task, "mode"):
      return

    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    head_pos = self.interface.get_object_xpos_by_name(self.task._head_body_name, "OBJ_BODY")
    arrow_pos = [head_pos[0], head_pos[1], head_pos[2] + 0.5]

    root_quat = self.interface.get_object_xquat_by_name(self.task._root_body_name, "OBJ_BODY")
    root_yaw = tf3.euler.quat2euler(root_quat)[2]

    mode = self.task.mode
    yaw_ref, vx_ref, vy_ref = self.task.mode_ref
    rgba_blue = np.array([0, 0, 1, 0.5])

    if mode == cmd_walking_task.CmdWalkModes.FORWARD:
      length = float(np.linalg.norm([vx_ref, vy_ref]))
      mat = tf3.euler.euler2mat(0, np.pi / 2, root_yaw)
    elif mode == cmd_walking_task.CmdWalkModes.STANDING:
      length = 0.0
      mat = tf3.euler.euler2mat(0, 0, 0)
    else:
      length = 0.0
      mat = tf3.euler.euler2mat(0, np.pi, 0)
    marker_drawer.add_marker(pos=arrow_pos, mat=mat, size=[0.05, 0.05, 2 * length], rgba=rgba_blue, type=arrow)

  # --- interactive command interface ---
  KEY_COMMANDS = {264: "STANDING", 265: "FORWARD", 263: "TURN_LEFT", 262: "TURN_RIGHT"}
  KEY_REFS = {"STANDING": [0, 0, 0], "FORWARD": [0, 0.2, 0], "TURN_LEFT": [0.4, 0, 0], "TURN_RIGHT": [-0.4, 0, 0]}

  def apply_command(self, name):
    self.task.mode = getattr(cmd_walking_task.CmdWalkModes, name)
    self.task.mode_ref = np.array(self.KEY_REFS[name])
    self.sticky_command = name

  def on_key_command(self, keycode):
    if keycode in self.KEY_COMMANDS:
      self.task.random_mode_switch = False
      self.apply_command(self.KEY_COMMANDS[keycode])
