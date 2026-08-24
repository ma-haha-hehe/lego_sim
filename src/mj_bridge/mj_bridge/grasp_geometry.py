"""MuJoCo geometry helpers shared by the bridge and pure-Python tests."""

from __future__ import annotations

import mujoco
import numpy as np


COLLISION_Z_OFFSET = -0.009
PRIMARY_PAD_CENTER_Z_FROM_FINGER = 0.0445


def grasp_center_world(model, data) -> np.ndarray | None:
    """Return the midpoint of the two primary fingertip contact pads."""
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    if min(hand_id, left_id, right_id) < 0:
        return None
    finger_midpoint = (data.xpos[left_id] + data.xpos[right_id]) / 2.0
    hand_rotation = data.xmat[hand_id].reshape(3, 3)
    return finger_midpoint + hand_rotation @ np.array(
        [0.0, 0.0, PRIMARY_PAD_CENTER_Z_FROM_FINGER]
    )


def block_collision_center_world(data, body_id: int) -> np.ndarray:
    """Return the center of the main box collider for a benchmark block."""
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ np.array([0.0, 0.0, COLLISION_Z_OFFSET])


def align_block_to_grasp_center(model, data, body_id: int) -> float:
    """Move a free block so its physical center lies between the finger pads."""
    center = grasp_center_world(model, data)
    if center is None:
        return float("inf")
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0:
        return float("inf")
    qadr = model.jnt_qposadr[joint_id]
    dofadr = model.jnt_dofadr[joint_id]
    old_center = block_collision_center_world(data, body_id).copy()
    rotation = data.xmat[body_id].reshape(3, 3)
    body_position = center - rotation @ np.array([0.0, 0.0, COLLISION_Z_OFFSET])
    data.qpos[qadr:qadr + 3] = body_position
    data.qvel[dofadr:dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(old_center - center))
