"""Environment registry for learninghumanoidwalking.

All environment classes should be imported and registered here.
This makes environments discoverable for testing and external use.
"""

from envs.cartpole import CartpoleEnv
from envs.g1 import G1CmdWalkEnv
from envs.h1 import H1CmdWalkEnv, H1Env, H1WalkEnv
from envs.jvrc import JvrcStepEnv, JvrcWalkEnv

# Registry of all available environments
# Maps environment name -> (class, robot_name)
ENVIRONMENTS = {
    "jvrc_walk": (JvrcWalkEnv, "jvrc"),
    "jvrc_step": (JvrcStepEnv, "jvrc"),
    "h1": (H1Env, "h1"),
    "h1_walk": (H1WalkEnv, "h1"),
    "cartpole": (CartpoleEnv, "cartpole"),
    "h1_cmd_walk": (H1CmdWalkEnv, "h1"),
    "g1_cmd_walk": (G1CmdWalkEnv, "g1"),
}

__all__ = [
    "JvrcWalkEnv",
    "JvrcStepEnv",
    "H1Env",
    "H1WalkEnv",
    "H1CmdWalkEnv",
    "G1CmdWalkEnv",
    "CartpoleEnv",
    "ENVIRONMENTS",
]
