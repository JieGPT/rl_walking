import os

from dm_control import mjcf

import models

G1_DESCRIPTION_PATH = os.path.join(os.path.dirname(models.__file__), "mujoco_menagerie/unitree_g1/scene.xml")

# Per-leg order follows the model's actuator order
# (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll) so that PD
# gains and action vectors align with data.ctrl. Sign-flipped DOFs under
# left-right mirroring: hip_roll, hip_yaw, ankle_roll.
LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def remove_joints_and_actuators(mjcf_model, config):
    # remove joints
    for limb in config["unused_joints"]:
        for joint in limb:
            mjcf_model.find("joint", joint).remove()

    # remove all actuators with no corresponding joint
    for act in list(mjcf_model.actuator.position):
        if act.joint is None:
            act.remove()
    return mjcf_model


def position_to_motor_actuators(mjcf_model):
    """Convert position actuators to motor (torque) actuators.

    The menagerie G1 model ships with position actuators, but the torque-based
    PD control in robots/robot_base.py requires motor actuators (ctrl = joint
    torque, gear 1). Actuators are renamed to the "<joint>_motor" convention
    expected by RobotInterface.
    """
    for act in list(mjcf_model.actuator.position):
        joint, name = act.joint, act.name
        act.remove()
        mjcf_model.actuator.add("motor", joint=joint, name=name + "_motor")
    return mjcf_model


def builder(export_path, config):
    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(G1_DESCRIPTION_PATH)

    mjcf_model.model = "g1"

    # modify model
    mjcf_model = remove_joints_and_actuators(mjcf_model, config)
    mjcf_model = position_to_motor_actuators(mjcf_model)
    mjcf_model.find("default", "visual").geom.group = 1
    mjcf_model.find("default", "collision").geom.group = 2
    if "ctrllimited" in config:
        for act in mjcf_model.actuator.motor:
            act.ctrllimited = config["ctrllimited"]
    if "jointlimited" in config:
        mjcf_model.find("default", "g1").joint.limited = config["jointlimited"]

    # keyframe references the removed joints
    mjcf_model.keyframe.remove()

    # remove visual geoms, if needed.
    # NOTE: unlike the H1 model (primitive collision geoms), the G1 uses mesh
    # collision geoms, so the mesh assets must be kept.
    if "minimal" in config:
        if config["minimal"]:
            mjcf_model.find("default", "collision").geom.group = 1
            for geom in mjcf_model.find_all("geom"):
                if geom.dclass:
                    if geom.dclass.dclass == "visual":
                        geom.remove()

    # set name of freejoint
    mjcf_model.find("body", "pelvis").freejoint.name = "root"

    # set some size options
    mjcf_model.size.njmax = "-1"
    mjcf_model.size.nconmax = "-1"
    mjcf_model.size.nuser_actuator = "-1"

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + ".xml")
    print("Exporting XML model to ", path_to_xml)
    return
