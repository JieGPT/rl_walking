"""Command-following walking task.

4-mode command (STANDING / FORWARD / TURN_LEFT / TURN_RIGHT) with a 3-D
velocity reference [yaw_vel, vx, vy]. Only the commanded component is
tracked: FORWARD tracks forward velocity, the turn modes track yaw rate
while staying in place.
"""

from enum import Enum, auto

import numpy as np

from tasks.walking_task import WalkingTask


class CmdWalkModes(Enum):
  STANDING = auto()
  FORWARD = auto()
  TURN_LEFT = auto()
  TURN_RIGHT = auto()

  def encode(self):
    # One-hot encoding; explicit index lookup so the encoding does not
    # depend on the auto() values.
    vec = np.zeros(len(CmdWalkModes))
    vec[list(CmdWalkModes).index(self)] = 1.0
    return vec

  def sample_ref(self):
    if self is CmdWalkModes.STANDING:
      return np.zeros(3)
    if self is CmdWalkModes.FORWARD:
      return np.array([0.0, np.random.uniform(0.1, 0.3), 0.0]) # [0, +v, 0]
    omega = np.random.uniform(0.3, 0.6)
    return np.array([omega, 0.0, 0.0]) if self is CmdWalkModes.TURN_LEFT else np.array([-omega, 0.0, 0.0])


class CmdWalkingTask(WalkingTask):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.random_mode_switch = True

  @property
  def is_standing(self):
    return self.mode == CmdWalkModes.STANDING

  def _decompose_mode_ref(self):
    yaw, vx, vy = self.mode_ref
    if self.mode == CmdWalkModes.STANDING:
      return 0.0, 0.0, 0.0
    if self.mode == CmdWalkModes.FORWARD:
      return 0.0, vx, 0.0
    return yaw, 0.0, 0.0  # turning: track yaw, stay in place

  def _init_mode(self):
    self.mode = np.random.choice(list(CmdWalkModes), p=[0.35, 0.3, 0.175, 0.175])
    self.mode_ref = self.mode.sample_ref()

  def _switch_mode(self):
    if not self.random_mode_switch:
      return
    if np.random.randint(100) != 0:
      return
    # Leaving STANDING is always safe (both feet grounded); a mid-gait
    # switch is only allowed during double support.
    if not self.is_standing:
      in_double_support = self.right_clock[0](self._phase) == 1 and self.left_clock[0](self._phase) == 1
      if not in_double_support:
        return
    others = [m for m in CmdWalkModes if m is not self.mode]
    self.mode = np.random.choice(others)
    self.mode_ref = self.mode.sample_ref()
