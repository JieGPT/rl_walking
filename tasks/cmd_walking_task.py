from enum import Enum, auto
import numpy as np

from tasks import rewards
from tasks.walking_task import WalkingTask

class CmdWalkModes(Enum):
  STANDING = auto()
  FORWARD = auto()
  TURN_LEFT = auto()
  TURN_RIGHT = auto()

  def encode(self):
    return np.eye(4)[self.value -1]
  
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
  
  def step(self):
    super().step()  # phase advance + hfield manipulation
    if not self.random_mode_switch:
      return
    if np.random.randint(100) == 0:
      others = [m for m in CmdWalkModes if m is not self.mode]
      self.mode = np.random.choice(others)
      self.mode_ref = self.mode.sample_ref()

  def reset(self, iter_count=0):
    self.mode = np.random.choice(list(CmdWalkModes), p=[0.35, 0.3, 0.175, 0.175])
    self.mode_ref = self.mode.sample_ref()
    
    # gait clocks + phase: reuse base logic
    self.right_clock, self.left_clock = rewards.create_phase_reward(
      self._swing_duration, self._stance_duration, 0.1, "grounded", 1 / self._control_dt
    )

    # number of control steps in one full cycle
    # (one full cycle includes left swing + right swing)
    self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
    self._phase = np.random.randint(0, self._period)

  