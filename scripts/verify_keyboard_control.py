"""Quick verification of the H1CmdWalkEnv keyboard command interface.

Launches the MuJoCo viewer and steps the robot with a (possibly untrained)
policy. No training is required: the current mode and reference velocity are
printed on every change, which is enough to verify the key -> command
plumbing. The robot will fall with the default zero-action policy, which is
fine; pass a trained actor with --actor to watch it follow commands.

Key bindings (GLFW arrow keys, see H1CmdWalkEnv.KEY_COMMANDS):
    DOWN  (264) -> STANDING
    UP    (265) -> FORWARD
    LEFT  (263) -> TURN_LEFT
    RIGHT (262) -> TURN_RIGHT
    'P' -> pause, close the window to quit

Usage:
    uv run scripts/verify_keyboard_control.py
    uv run scripts/verify_keyboard_control.py --actor path/to/actor.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENVS = {
    "h1": ("envs.h1", "H1CmdWalkEnv"),
    "g1": ("envs.g1", "G1CmdWalkEnv"),
}


def main():
    parser = argparse.ArgumentParser(description="Verify cmd-walk env keyboard control")
    parser.add_argument("--env", type=str, default="h1", choices=list(ENVS), help="Robot environment to verify")
    parser.add_argument("--actor", type=Path, default=None, help="Path to a trained actor_*.pt (optional)")
    args = parser.parse_args()

    import importlib

    module = importlib.import_module(ENVS[args.env][0])
    EnvClass = getattr(module, ENVS[args.env][1])
    env = EnvClass()

    policy = None
    if args.actor is not None:
        policy = torch.load(args.actor, weights_only=False)
        policy.eval()

    obs = env.reset()
    env.render()
    print("Keys: DOWN -> STANDING | UP -> FORWARD | LEFT -> TURN_LEFT | RIGHT -> TURN_RIGHT | P -> pause")
    print(f"Initial mode: {env.task.mode.name}, ref={np.round(env.task.mode_ref, 2)}")

    last_mode = None
    while env.viewer.is_running():
        if policy is not None:
            with torch.no_grad():
                action = policy.forward(torch.tensor(obs, dtype=torch.float32), deterministic=True).numpy()
        else:
            action = np.zeros(env.action_space.shape[0])

        obs, _, done, _ = env.step(action)

        if done and hasattr(env, "sticky_command"):
            env.apply_command(env.sticky_command)

        if env.task.mode is not last_mode:
            print(
                f"mode={env.task.mode.name:10s} ref={np.round(env.task.mode_ref, 2)} "
                f"random_switch={env.task.random_mode_switch}"
            )
            last_mode = env.task.mode

        env.render()

    env.close()


if __name__ == "__main__":
    main()
