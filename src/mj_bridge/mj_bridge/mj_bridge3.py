#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import threading
import json
import math
import numpy as np

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .grasp_geometry import (
    COLLISION_Z_OFFSET, block_collision_center_world, grasp_center_world,
)

BASE = os.path.dirname(__file__)
sys.path.append(BASE)

from scene_builder import build
from benchmark_core import dump_json, load_yaml as load_benchmark_yaml, score_episode

def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    return np.array(
        [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ],
        dtype=float,
    )


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion in wxyz order."""
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in wxyz order."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dtype=float)


def snap_yaw_to_90(yaw: float) -> float:
    return round(yaw / (np.pi / 2.0)) * (np.pi / 2.0)


def snap_xy_to_stud_grid(x: float, y: float) -> tuple[float, float]:
    local_x = x - ASSEMBLY_BASE_CENTER_X
    local_y = y - ASSEMBLY_BASE_CENTER_Y

    snapped_local_x = round(local_x / STUD_PITCH) * STUD_PITCH
    snapped_local_y = round(local_y / STUD_PITCH) * STUD_PITCH

    return (
        ASSEMBLY_BASE_CENTER_X + snapped_local_x,
        ASSEMBLY_BASE_CENTER_Y + snapped_local_y,
    )

def infer_brick_type_from_name(name: str):
    if "2x2" in name:
        return "brick_2x2"
    if "4x2" in name or "2x4" in name:
        return "brick_4x2"
    return None


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Return yaw from a MuJoCo quaternion in w, x, y, z order."""
    w, x, y, z = q
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def get_world_half_size(body_name: str, quat_wxyz: np.ndarray):
    """Return the world-frame XY half extents for a brick footprint."""
    brick_type = infer_brick_type_from_name(body_name)

    if brick_type not in BRICK_HALF_SIZE:
        return None

    hx, hy, hz = BRICK_HALF_SIZE[brick_type]

    yaw = yaw_from_quat_wxyz(quat_wxyz)

    c = abs(np.cos(yaw))
    s = abs(np.sin(yaw))

    world_hx = c * hx + s * hy
    world_hy = s * hx + c * hy

    return world_hx, world_hy, hz


def xy_overlap_area(pos_a, half_a, pos_b, half_b) -> float:
    ax1 = pos_a[0] - half_a[0]
    ax2 = pos_a[0] + half_a[0]
    ay1 = pos_a[1] - half_a[1]
    ay2 = pos_a[1] + half_a[1]

    bx1 = pos_b[0] - half_b[0]
    bx2 = pos_b[0] + half_b[0]
    by1 = pos_b[1] - half_b[1]
    by2 = pos_b[1] + half_b[1]

    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0.0, min(ay2, by2) - max(ay1, by1))

    return overlap_x * overlap_y

# ============================================================

# ============================================================

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML_PATH = os.environ.get("MJ_BRIDGE_MODEL", os.path.join(SOURCE_DIR, "scene.xml"))
YAML_CONFIG_PATH = os.environ.get(
    "MJ_BRIDGE_INITIAL_POSITIONS",
    os.path.join(SOURCE_DIR, "initial_positions.yaml"),
)
BENCHMARK_CONFIG_PATH = os.environ.get(
    "MJ_BRIDGE_BENCHMARK_CONFIG",
    os.path.join(SOURCE_DIR, "benchmark.yaml"),
)
EPISODE_MANIFEST_PATH = os.environ.get("LEGO_BENCH_MANIFEST", "")
RUN_DIR = os.environ.get("LEGO_BENCH_RUN_DIR", "")


# ============================================================

# ============================================================

SIM_SUBSTEPS = 8
LOOP_DT = 0.016
GRASP_CONTACT_HOLD_S = 0.05
GRASP_AIRBORNE_SETTLE_S = 0.20
GRASP_MAX_TOOL_TILT_DEG = 3.0
ARM_GOAL_TOLERANCE_RAD = 0.010


# ============================================================

# ============================================================

GRIPPER_OPEN_VALUE = 0.04


GRIPPER_CLOSE_VALUE = 0.0



GRIPPER_ACTION_EXTRA_SLEEP = 0.02

# ============================================================

# ============================================================

BRICK_HALF_SIZE = {
    "brick_2x2": (0.016, 0.016, 0.0095),
    "brick_4x2": (0.032, 0.016, 0.0095),
}

MIN_OVERLAP_RATIO = 0.25


VERTICAL_CONTACT_TOL = 0.006

# ============================================================

# ============================================================

ASSEMBLY_BASE_NAME = "assembly_base_plate"

ASSEMBLY_BASE_CENTER_X = 0.35
ASSEMBLY_BASE_CENTER_Y = 0.35

STUD_PITCH = 0.016

BRICK_BODY_HALF_HEIGHT = 0.0095
TABLE_TOP_Z = 0.04
BASE_PLATE_THICKNESS = 0.006
BASE_STUD_HEIGHT = 0.004

BASE_PLATE_TOP_Z = TABLE_TOP_Z + BASE_PLATE_THICKNESS
BRICK_ON_BASE_CENTER_Z = BASE_PLATE_TOP_Z + BRICK_BODY_HALF_HEIGHT - COLLISION_Z_OFFSET

BASE_PLATE_HALF_X = 12 * STUD_PITCH / 2.0
BASE_PLATE_HALF_Y = 12 * STUD_PITCH / 2.0

BASE_SNAP_VERTICAL_TOL = 0.012


class MuJoCoActionServer(Node):
    def __init__(self):
        super().__init__("mj_action_server_node")
        self.get_logger().info("Starting MuJoCo action server")

        # ====================================================

        # ====================================================

        # The public CLI has already generated an episode-specific scene.
        if not EPISODE_MANIFEST_PATH:
            build()

        self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.mj_lock = threading.RLock()
        self.simulation_loop_ready = threading.Event()

        # ====================================================

        # ====================================================

        self.welded_pairs = set()
        self.fake_welds = []
        self.grasped_block = None

        # ====================================================

        # ====================================================

        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        self.finger_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        self.all_joint_names = self.arm_joint_names + self.finger_joint_names

        self.joint_qpos_addr = {}
        self.joint_qvel_addr = {}

        for name in self.all_joint_names:
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if jid == -1:
                self.get_logger().warn(f"Joint not found in the MuJoCo model: {name}")
                continue

            self.joint_qpos_addr[name] = self.model.jnt_qposadr[jid]
            self.joint_qvel_addr[name] = self.model.jnt_dofadr[jid]

        # ====================================================

        # ====================================================

        self.target_qpos = np.zeros(self.model.nq)
        self.target_qvel = np.zeros(self.model.nv)

        self.current_goal_handle = None
        self.trajectory = None
        self.active_joint_names = []

        self.traj_start_sim_time = 0.0
        self.traj_duration = 0.0
        self._last_arm_settle_log = 0.0
        self.is_executing = False
        self.cancel_requested = False

        # ====================================================

        # ====================================================


        self.gripper_target = GRIPPER_OPEN_VALUE


        self.gripper_start_value = GRIPPER_OPEN_VALUE


        self.gripper_goal_value = GRIPPER_OPEN_VALUE


        self.gripper_start_sim_time = 0.0


        self.gripper_duration = 1.0


        self.gripper_moving = False

        # open / close_hold
        self.gripper_mode = "open"
        self.grasp_contact_body = None
        self.grasp_contact_since = None
        self._last_grasp_wait_log = 0.0
        self.grasp_reference_offset = None
        self.grasp_reference_rotation = None
        self.active_grasp_max_translation_slip = 0.0
        self.active_grasp_translation_delta_at_max = np.zeros(3, dtype=float)
        self.active_grasp_max_rotation_slip_deg = 0.0
        self.active_grasp_last_logged_slip = 0.0
        self.grasp_airborne_started = False
        self.grasp_airborne_finished = False
        self.grasp_airborne_since = None
        self.grasp_translation_slip_by_body = {}
        self.grasp_rotation_slip_by_body = {}

        # ====================================================

        # ====================================================

        self.load_initial_positions()

        with self.mj_lock:
            self.target_qpos[:] = self.data.qpos[:]
            self.target_qvel[:] = 0.0


        self.episode_initial_qpos = self.data.qpos.copy()
        self.episode_initial_qvel = self.data.qvel.copy()
        self.benchmark_config = self.load_benchmark_config()
        self.episode_manifest = (
            load_benchmark_yaml(EPISODE_MANIFEST_PATH)
            if EPISODE_MANIFEST_PATH and os.path.exists(EPISODE_MANIFEST_PATH)
            else None
        )
        self.reset_benchmark_state()

        # ====================================================

        # ====================================================

        self.joint_state_pub = self.create_publisher(
            JointState,
            "/joint_states",
            20,
        )

        self.joint_state_timer = self.create_timer(
            0.02,
            self.publish_joint_states,
        )

        self.benchmark_pub = self.create_publisher(String, "/mj_bridge/benchmark_state", 10)
        self.goal_pub = self.create_publisher(String, "/lego_bench/goal", 10)
        self.observation_mode = os.environ.get("LEGO_BENCH_OBSERVATION", "oracle")
        self.oracle_pub = (
            self.create_publisher(String, "/lego_bench/ground_truth", 10)
            if self.observation_mode == "oracle" else None
        )
        self.benchmark_timer = self.create_timer(0.05, self.update_benchmark_state)
        self.reset_service = self.create_service(Trigger, "/mj_bridge/reset", self.handle_reset)
        self.result_service = self.create_service(Trigger, "/mj_bridge/result", self.handle_result)
        self.connection_mode = os.environ.get("LEGO_BENCH_CONNECTION_MODE", "physics")
        self.camera_renderer = None
        self.camera_on_demand = os.environ.get(
            "LEGO_BENCH_CAMERA_ON_DEMAND", "false"
        ).lower() in ("1", "true", "yes")
        self.camera_request_lock = threading.Lock()
        self.camera_capture_requested = 0
        self.camera_capture_completed = 0
        if self.observation_mode == "rgbd":
            self.init_virtual_camera()

        # ====================================================
        # 8. Action Server
        # ====================================================

        self._arm_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/mj_panda_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_arm_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            handle_accepted_callback=self.handle_arm_accepted_callback,
        )

        self._hand_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/mj_panda_hand_controller/follow_joint_trajectory",
            execute_callback=self.execute_hand_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("Arm and gripper action servers are ready")
        self.get_logger().info("Arm action: /mj_panda_arm_controller/follow_joint_trajectory")
        self.get_logger().info("Hand action: /mj_panda_hand_controller/follow_joint_trajectory")
        self.get_logger().info("Benchmark services: /mj_bridge/reset, /mj_bridge/result")
        if self.episode_manifest:
            self.get_logger().info(
                f"LEGO Bench episode: {self.episode_manifest['episode_id']} "
                f"({len(self.episode_manifest['spawned_blocks'])} parts)"
            )
            self.get_logger().info(f"Connection mode: {self.connection_mode}")

    def init_virtual_camera(self):
        self.camera_width = 960
        self.camera_height = 720
        self.camera_fovy = 45.0
        self.camera_renderer = mujoco.Renderer(
            self.model, height=self.camera_height, width=self.camera_width
        )
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", 5)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 5)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self.capture_service = self.create_service(
            Trigger, "/mj_bridge/capture_rgbd", self.handle_capture_rgbd
        )
        self._last_camera_publish = 0.0
        mode = "on demand" if self.camera_on_demand else "continuous"
        self.get_logger().info(f"RGB-D observation enabled on /camera/* ({mode})")

    def handle_capture_rgbd(self, request, response):
        """Queue one RGB-D frame without rendering in the ROS callback thread."""
        del request
        if self.camera_renderer is None:
            response.success = False
            response.message = "RGB-D observation is disabled"
            return response
        with self.camera_request_lock:
            self.camera_capture_requested += 1
            capture_id = self.camera_capture_requested
        response.success = True
        response.message = f"RGB-D capture {capture_id} queued"
        return response

    def pending_camera_capture(self):
        """Return the newest pending capture ID when the robot is stationary."""
        with self.camera_request_lock:
            if self.camera_capture_requested <= self.camera_capture_completed:
                return None
            capture_id = self.camera_capture_requested
        with self.mj_lock:
            if self.is_executing or self.gripper_moving:
                return None
        return capture_id

    def _image_message(self, array, encoding):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "realsense"
        msg.height, msg.width = array.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(array.strides[0])
        msg.data = np.ascontiguousarray(array).tobytes()
        return msg

    def publish_virtual_camera(self, capture_id=None):
        if self.camera_renderer is None:
            return
        started = time.perf_counter()
        with self.mj_lock:
            self.camera_renderer.update_scene(self.data, camera="realsense")
            rgb = self.camera_renderer.render().copy()
            self.camera_renderer.enable_depth_rendering()
            self.camera_renderer.update_scene(self.data, camera="realsense")
            depth = self.camera_renderer.render().astype(np.float32).copy()
            self.camera_renderer.disable_depth_rendering()
        self.rgb_pub.publish(self._image_message(rgb, "rgb8"))
        self.depth_pub.publish(self._image_message(depth, "32FC1"))
        info = CameraInfo()
        info.header.stamp = self.get_clock().now().to_msg()
        info.header.frame_id = "realsense"
        info.width, info.height = self.camera_width, self.camera_height
        fy = self.camera_height / (2.0 * math.tan(math.radians(self.camera_fovy) / 2.0))
        fx = fy
        cx, cy = self.camera_width / 2.0, self.camera_height / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        self.camera_info_pub.publish(info)
        if capture_id is not None:
            with self.camera_request_lock:
                self.camera_capture_completed = max(
                    self.camera_capture_completed, capture_id
                )
            self.get_logger().info(
                f"Published requested RGB-D frame {capture_id} in "
                f"{time.perf_counter() - started:.3f}s"
            )

    def load_benchmark_config(self):
        defaults = {
            "target_body": "2x2_brick_1",
            "lift_height_m": 0.05,
            "hold_time_s": 0.5,
            "max_episode_time_s": 30.0,
            "grasp_translation_slip_tolerance_m": 0.002,
            "grasp_rotation_slip_tolerance_deg": 3.0,
        }
        if not os.path.exists(BENCHMARK_CONFIG_PATH):
            self.get_logger().warn(f"Benchmark configuration not found: {BENCHMARK_CONFIG_PATH}")
            return defaults
        with open(BENCHMARK_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
        return defaults

    def reset_benchmark_state(self):
        self.collision_count = 0
        self.stability_violations = 0
        self._active_safety_contacts = set()
        self._unstable_blocks = set()
        self.grasp_contact_body = None
        self.grasp_contact_since = None
        self._last_grasp_wait_log = 0.0
        self.grasp_reference_offset = None
        self.grasp_reference_rotation = None
        self.active_grasp_max_translation_slip = 0.0
        self.active_grasp_translation_delta_at_max = np.zeros(3, dtype=float)
        self.active_grasp_max_rotation_slip_deg = 0.0
        self.active_grasp_last_logged_slip = 0.0
        self.grasp_airborne_started = False
        self.grasp_airborne_finished = False
        self.grasp_airborne_since = None
        self.grasp_translation_slip_by_body = {}
        self.grasp_rotation_slip_by_body = {}
        if self.episode_manifest:
            self.target_body_id = -1
            self.episode_start_time = time.monotonic()
            self.lift_start_time = None
            self.episode_success = False
            self.episode_timeout = False
            self.initial_target_z = None
            return
        target = str(self.benchmark_config["target_body"])
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target)
        self.target_body_id = body_id
        self.episode_start_time = time.monotonic()
        self.lift_start_time = None
        self.episode_success = False
        self.episode_timeout = False
        self.initial_target_z = float(self.data.xpos[body_id][2]) if body_id >= 0 else None

    def benchmark_result(self):
        with self.mj_lock:
            if self.episode_manifest:
                current_state = self.current_block_state()
                scored = score_episode(
                    self.episode_manifest,
                    current_state,
                    xy_tol=float(self.benchmark_config.get("position_tolerance_xy_m", 0.006)),
                    z_tol=float(self.benchmark_config.get("position_tolerance_z_m", 0.004)),
                    yaw_tol_deg=float(self.benchmark_config.get("yaw_tolerance_deg", 8.0)),
                )
                max_translation_slip = max(
                    self.grasp_translation_slip_by_body.values(), default=0.0
                )
                max_rotation_slip = max(
                    self.grasp_rotation_slip_by_body.values(), default=0.0
                )
                grasp_stable = (
                    max_translation_slip <= float(self.benchmark_config.get(
                        "grasp_translation_slip_tolerance_m", 0.002
                    )) and
                    max_rotation_slip <= float(self.benchmark_config.get(
                        "grasp_rotation_slip_tolerance_deg", 3.0
                    ))
                )
                scored["success"] = bool(scored["success"] and grasp_stable)
                scored.update({
                    "episode_id": self.episode_manifest["episode_id"],
                    "seed": self.episode_manifest["seed"],
                    "elapsed_s": time.monotonic() - self.episode_start_time,
                    "timeout": self.episode_timeout,
                    "collision_count": self.collision_count,
                    "stability_violations": self.stability_violations,
                    "observation_mode": self.observation_mode,
                    "connection_mode": self.connection_mode,
                    "grasped_block": current_state["grasped_block"],
                    "grasp_stable": grasp_stable,
                    "max_grasp_translation_slip_m": max_translation_slip,
                    "max_grasp_rotation_slip_deg": max_rotation_slip,
                    "grasp_slip_by_block": {
                        item["id"]: {
                            "translation_m": self.grasp_translation_slip_by_body.get(
                                item["body_name"], 0.0
                            ),
                            "rotation_deg": self.grasp_rotation_slip_by_body.get(
                                item["body_name"], 0.0
                            ),
                        }
                        for item in self.episode_manifest["spawned_blocks"]
                    },
                })
                return scored
            current_z = None
            if self.target_body_id >= 0:
                current_z = float(self.data.xpos[self.target_body_id][2])
            return {
                "target_body": self.benchmark_config["target_body"],
                "success": self.episode_success,
                "timeout": self.episode_timeout,
                "elapsed_s": time.monotonic() - self.episode_start_time,
                "initial_z_m": self.initial_target_z,
                "current_z_m": current_z,
                "required_lift_m": float(self.benchmark_config["lift_height_m"]),
                "required_hold_s": float(self.benchmark_config["hold_time_s"]),
                "collision_count": self.collision_count,
                "stability_violations": self.stability_violations,
            }

    def update_safety_metrics(self):
        """Count contact/instability transitions, rather than every simulation step."""
        if not self.episode_manifest:
            return
        with self.mj_lock:
            contacts = set()
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                body1 = int(self.model.geom_bodyid[contact.geom1])
                body2 = int(self.model.geom_bodyid[contact.geom2])
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
                names = {name1, name2}
                robot_contact = any(name.startswith("panda_") or name == "hand" for name in names)
                environment_contact = any(
                    name in {"table", "assembly_base_plate", "world"} for name in names
                )
                if robot_contact and environment_contact:
                    contacts.add(tuple(sorted(names)))
            self.collision_count += len(contacts - self._active_safety_contacts)
            self._active_safety_contacts = contacts

            unstable = set()
            for item in self.episode_manifest["spawned_blocks"]:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"])
                if body_id < 0:
                    continue
                x, y, z = (float(v) for v in self.data.xpos[body_id])
                # Local +Z projected on world +Z. Below 0.7 means a severe tilt (>45 degrees).
                upright = float(self.data.xmat[body_id][8]) >= 0.7
                if z < 0.025 or not (-0.75 <= x <= 0.75 and -0.75 <= y <= 0.75) or not upright:
                    unstable.add(item["id"])
            new_violations = unstable - self._unstable_blocks
            self.stability_violations += len(new_violations)
            for block_id in sorted(new_violations):
                item = next(
                    block for block in self.episode_manifest["spawned_blocks"]
                    if block["id"] == block_id
                )
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
                )
                x, y, z = (float(value) for value in self.data.xpos[body_id])
                upright_z = float(self.data.xmat[body_id][8])
                self.get_logger().warn(
                    f"[STABILITY_VIOLATION] {block_id}, "
                    f"position=({x:.4f}, {y:.4f}, {z:.4f}), upright_z={upright_z:.4f}"
                )
            self._unstable_blocks = unstable

    def update_benchmark_state(self):
        if self.episode_manifest:
            self.episode_timeout = time.monotonic() - self.episode_start_time >= float(
                self.benchmark_config["max_episode_time_s"]
            )
            result = self.benchmark_result()
            self.episode_success = bool(result["success"])
            self.publish_public_state(result)
            return
        if self.target_body_id < 0 or self.episode_success or self.episode_timeout:
            return
        with self.mj_lock:
            now = time.monotonic()
            current_z = float(self.data.xpos[self.target_body_id][2])
            lifted = current_z - self.initial_target_z >= float(self.benchmark_config["lift_height_m"])
            if lifted:
                if self.lift_start_time is None:
                    self.lift_start_time = now
                elif now - self.lift_start_time >= float(self.benchmark_config["hold_time_s"]):
                    self.episode_success = True
            else:
                self.lift_start_time = None
            self.episode_timeout = now - self.episode_start_time >= float(
                self.benchmark_config["max_episode_time_s"]
            )
        msg = String()
        msg.data = json.dumps(self.benchmark_result(), ensure_ascii=False)
        self.benchmark_pub.publish(msg)

    def current_block_state(self):
        """Return the authoritative MuJoCo state keyed by public product block ID."""
        blocks = {}
        if not self.episode_manifest:
            return {"blocks": blocks}
        grasped_id = None
        for item in self.episode_manifest["spawned_blocks"]:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
            )
            if body_id < 0:
                continue
            blocks[item["id"]] = {
                "type": item["type"],
                "color": item["color"],
                "body_name": item["body_name"],
                "position": [float(v) for v in self.data.xpos[body_id]],
                "quaternion_wxyz": [float(v) for v in self.data.xquat[body_id]],
                "yaw_rad": float(yaw_from_quat_wxyz(self.data.xquat[body_id])),
            }
            if item["body_name"] == self.grasped_block:
                grasped_id = item["id"]
        return {
            "episode_id": self.episode_manifest["episode_id"],
            "grasped_block": grasped_id,
            "blocks": blocks,
        }

    def publish_public_state(self, result=None):
        state = self.current_block_state()
        if self.oracle_pub is not None:
            oracle = String()
            oracle.data = json.dumps(state, ensure_ascii=False)
            self.oracle_pub.publish(oracle)
        goal = String()
        goal.data = json.dumps({
            "episode_id": self.episode_manifest["episode_id"],
            "product": self.episode_manifest["product"],
            "target_blocks": self.episode_manifest["target_blocks"],
        }, ensure_ascii=False)
        self.goal_pub.publish(goal)
        msg = String()
        msg.data = json.dumps(result or self.benchmark_result(), ensure_ascii=False)
        self.benchmark_pub.publish(msg)

    def handle_reset(self, request, response):
        del request
        if not self.simulation_loop_ready.wait(timeout=30.0):
            response.success = False
            response.message = "Simulation loop did not become ready"
            return response
        with self.mj_lock:
            self.data.qpos[:] = self.episode_initial_qpos
            self.data.qvel[:] = self.episode_initial_qvel
            self.target_qpos[:] = self.episode_initial_qpos
            self.target_qvel[:] = 0.0
            self.trajectory = None
            self.is_executing = False
            self.gripper_target = GRIPPER_OPEN_VALUE
            self.gripper_goal_value = GRIPPER_OPEN_VALUE
            self.gripper_moving = False
            self.gripper_mode = "open"
            self.fake_welds.clear()
            self.welded_pairs.clear()
            self.grasped_block = None
            self.grasp_contact_body = None
            self.grasp_contact_since = None
            mujoco.mj_forward(self.model, self.data)
            self.reset_benchmark_state()
        response.success = self.episode_manifest is not None or self.target_body_id >= 0
        response.message = json.dumps(self.benchmark_result(), ensure_ascii=False)
        return response

    def handle_result(self, request, response):
        del request
        result = self.benchmark_result()
        if RUN_DIR:
            dump_json(os.path.join(RUN_DIR, "actual_state.json"), self.current_block_state())
            dump_json(os.path.join(RUN_DIR, "result.json"), result)
        response.success = bool(result["success"])
        response.message = json.dumps(result, ensure_ascii=False)
        return response

    # =========================================================

    # =========================================================

    def load_initial_positions(self):
        if not os.path.exists(YAML_CONFIG_PATH):
            self.get_logger().warn(f"Initial joint configuration not found: {YAML_CONFIG_PATH}")
            return

        with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        positions = config.get("initial_positions", {})

        with self.mj_lock:
            for j_name, j_val in positions.items():
                jid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    j_name,
                )

                if jid == -1:
                    self.get_logger().warn(f"Unknown joint in the initial configuration: {j_name}")
                    continue

                qadr = self.model.jnt_qposadr[jid]
                self.data.qpos[qadr] = float(j_val)
                self.target_qpos[qadr] = float(j_val)


            for name in self.finger_joint_names:
                if name in self.joint_qpos_addr:
                    qadr = self.joint_qpos_addr[name]
                    self.data.qpos[qadr] = GRIPPER_OPEN_VALUE
                    self.target_qpos[qadr] = GRIPPER_OPEN_VALUE

            mujoco.mj_forward(self.model, self.data)

        self.get_logger().info("Loaded initial joint configuration")

    # =========================================================

    # =========================================================

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Received trajectory cancellation request")

        with self.mj_lock:
            self.cancel_requested = True
            self.is_executing = False

        return CancelResponse.ACCEPT

    # =========================================================

    # =========================================================

    def handle_arm_accepted_callback(self, goal_handle):
        with self.mj_lock:
            if self.current_goal_handle is not None and self.current_goal_handle.is_active:
                self.current_goal_handle.abort()

            self.current_goal_handle = goal_handle

        goal_handle.execute()

    def execute_arm_callback(self, goal_handle):
        traj = goal_handle.request.trajectory

        if not traj.points:
            self.get_logger().error("Arm trajectory contains no points")
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        joint_names = list(traj.joint_names)

        for name in joint_names:
            if name not in self.joint_qpos_addr:
                self.get_logger().error(f"Joint not found in the MuJoCo model: {name}")
                goal_handle.abort()
                return FollowJointTrajectory.Result()

        last_point = traj.points[-1]

        duration = (
            last_point.time_from_start.sec
            + last_point.time_from_start.nanosec * 1e-9
        )

        if duration <= 0.0:
            duration = 0.1

        with self.mj_lock:
            self.trajectory = traj
            self.active_joint_names = joint_names
            self.traj_start_sim_time = float(self.data.time)
            self.traj_duration = duration
            self.is_executing = True
            self.cancel_requested = False

        self.get_logger().info(
            f"Received arm trajectory: joints={len(joint_names)}, "
            f"points={len(traj.points)}, duration={duration:.3f}s"
        )

        timeout = duration * 12.0 + 15.0
        start_wait = time.monotonic()

        while rclpy.ok():
            with self.mj_lock:
                executing = self.is_executing
                cancelled = self.cancel_requested

            if cancelled:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            if not executing:
                break

            if time.monotonic() - start_wait > timeout:
                self.get_logger().error(f"Arm trajectory timed out after {timeout:.3f}s")
                with self.mj_lock:
                    self.is_executing = False
                goal_handle.abort()
                return FollowJointTrajectory.Result()

            time.sleep(0.01)

        if goal_handle.is_active:
            goal_handle.succeed()
            self.get_logger().info("Arm trajectory completed successfully")

        return FollowJointTrajectory.Result()

    # =========================================================

    # =========================================================

    def execute_hand_callback(self, goal_handle):
        """Start a smooth gripper transition and wait for it to complete."""
        traj = goal_handle.request.trajectory

        if not traj.points:
            self.get_logger().error("Gripper trajectory contains no points")
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        point = traj.points[-1]

        if not point.positions:
            self.get_logger().error("Gripper trajectory contains no positions")
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        cmd_val = float(point.positions[0])
        cmd_val = max(0.0, min(0.04, cmd_val))

        duration = (
            point.time_from_start.sec
            + point.time_from_start.nanosec * 1e-9
        )

        if duration <= 0.0:
            duration = 1.0

        with self.mj_lock:

            self.gripper_start_value = self.gripper_target
            self.gripper_goal_value = cmd_val
            self.gripper_start_sim_time = float(self.data.time)
            self.gripper_duration = duration
            self.gripper_moving = True

            if cmd_val >= 0.03:
                self.gripper_mode = "opening"
                self.get_logger().info(
                    f"Opening gripper smoothly: {self.gripper_start_value:.4f} -> "
                    f"{self.gripper_goal_value:.4f}, duration={duration:.2f}s"
                )
            else:
                self.gripper_mode = "closing"
                self.get_logger().info(
                    f"Closing gripper smoothly: {self.gripper_start_value:.4f} -> "
                    f"{self.gripper_goal_value:.4f}, duration={duration:.2f}s"
                )



        start_wait = time.monotonic()
        timeout = duration * 12.0 + 5.0

        while rclpy.ok():
            with self.mj_lock:
                moving = self.gripper_moving

            if not moving:
                break

            if time.monotonic() - start_wait > timeout:
                self.get_logger().error("Gripper action timed out")
                goal_handle.abort()
                return FollowJointTrajectory.Result()

            time.sleep(0.01)

        time.sleep(GRIPPER_ACTION_EXTRA_SLEEP)

        if goal_handle.is_active:
            goal_handle.succeed()

        return FollowJointTrajectory.Result()

    def update_gripper_target(self):
        """Interpolate the gripper actuator target once per simulation frame."""
        with self.mj_lock:
            if not self.gripper_moving:
                return

            elapsed = float(self.data.time) - self.gripper_start_sim_time
            alpha = elapsed / self.gripper_duration
            alpha = max(0.0, min(1.0, alpha))

            self.gripper_target = (
                self.gripper_start_value
                + alpha * (self.gripper_goal_value - self.gripper_start_value)
            )

            if alpha >= 1.0:
                self.gripper_target = self.gripper_goal_value
                self.gripper_moving = False
                if self.gripper_goal_value >= 0.03:
                    self.gripper_mode = "open"
                    self.get_logger().info("Gripper opening completed")
                else:
                    self.gripper_mode = "close_hold"
                    self.get_logger().info("Gripper closing completed")

    # =========================================================

    # =========================================================

    def update_action_state(self):
        with self.mj_lock:
            if not self.is_executing or self.trajectory is None:
                return

            # A carried part regaining non-gripper support means the vertical
            # placement stroke has reached the product or base plate. Stop the
            # servo at first physical contact instead of forcing the arm
            # through the assembled geometry while it chases the nominal end
            # point.
            if self.grasped_block is not None and self.grasp_airborne_finished:
                for name in self.active_joint_names:
                    if name in self.joint_qpos_addr:
                        qadr = self.joint_qpos_addr[name]
                        self.target_qpos[qadr] = self.data.qpos[qadr]
                self.is_executing = False
                self.get_logger().info(
                    f"[PLACE_CONTACT_STOP] {self.grasped_block}, "
                    "support_contact=true"
                )
                return

            # Use simulated time so a loaded host cannot skip trajectory
            # segments and inject large position-target discontinuities.
            t_rel = float(self.data.time) - self.traj_start_sim_time
            points = self.trajectory.points

            if not points:
                self.is_executing = False
                return

            if t_rel >= self.traj_duration:
                self.set_target_direct_locked(points[-1])
                final_point = points[-1]
                errors = []
                for index, name in enumerate(self.active_joint_names):
                    if (name not in self.joint_qpos_addr or
                            index >= len(final_point.positions)):
                        continue
                    qadr = self.joint_qpos_addr[name]
                    errors.append(abs(
                        float(final_point.positions[index]) -
                        float(self.data.qpos[qadr])
                    ))
                if not errors or max(errors) <= ARM_GOAL_TOLERANCE_RAD:
                    self.is_executing = False
                else:
                    now = time.monotonic()
                    if now - self._last_arm_settle_log >= 1.0:
                        worst_index = int(np.argmax(errors))
                        self.get_logger().info(
                            "[ARM_SETTLING] max_joint_error="
                            f"{errors[worst_index]:.4f}rad, "
                            f"joint={self.active_joint_names[worst_index]}"
                        )
                        self._last_arm_settle_log = now
                return

            idx = 0

            for i in range(len(points) - 1):
                t_next = (
                    points[i + 1].time_from_start.sec
                    + points[i + 1].time_from_start.nanosec * 1e-9
                )

                if t_rel <= t_next:
                    idx = i
                    break

            p0 = points[idx]
            p1 = points[min(idx + 1, len(points) - 1)]

            t0 = p0.time_from_start.sec + p0.time_from_start.nanosec * 1e-9
            t1 = p1.time_from_start.sec + p1.time_from_start.nanosec * 1e-9

            if abs(t1 - t0) < 1e-9:
                alpha = 0.0
            else:
                alpha = (t_rel - t0) / (t1 - t0)
                alpha = max(0.0, min(1.0, alpha))

            for i, name in enumerate(self.active_joint_names):
                if name not in self.joint_qpos_addr:
                    continue

                if i >= len(p0.positions) or i >= len(p1.positions):
                    continue

                qadr = self.joint_qpos_addr[name]

                self.target_qpos[qadr] = (
                    p0.positions[i]
                    + alpha * (p1.positions[i] - p0.positions[i])
                )

    def set_target_direct_locked(self, point):
        for i, name in enumerate(self.active_joint_names):
            if name not in self.joint_qpos_addr:
                continue

            if i >= len(point.positions):
                continue

            qadr = self.joint_qpos_addr[name]
            self.target_qpos[qadr] = point.positions[i]

    # =========================================================

    # =========================================================

    def step_pid(self):
        with self.mj_lock:
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0

            # The hardware Panda controller supplies model-based gravity and
            # Coriolis feed-forward. Apply the MuJoCo equivalent only to the
            # seven arm joints; free-moving benchmark parts remain physical.
            for name in self.arm_joint_names:
                if name in self.joint_qvel_addr:
                    vadr = self.joint_qvel_addr[name]
                    self.data.qfrc_applied[vadr] = self.data.qfrc_bias[vadr]

            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    i,
                )

                if actuator_name is None:
                    continue

                jid = self.model.actuator_trnid[i, 0]
                qadr = self.model.jnt_qposadr[jid]
                if actuator_name.startswith("actuator"):
                    # Panda arm actuators are position servos in panda.xml.
                    # Supplying a torque here makes MuJoCo interpret it as an
                    # out-of-range joint target and causes visible oscillation.
                    self.data.ctrl[i] = self.target_qpos[qadr]


                elif actuator_name == "finger_actuator1":
                    self.data.ctrl[i] = self.gripper_target

                elif actuator_name == "finger_actuator2":
                    self.data.ctrl[i] = self.gripper_target

            mujoco.mj_step(self.model, self.data)

    def should_weld_bottom_to_top(self, upper_body_name: str, lower_body_name: str) -> bool:
        """Check whether an upper brick can engage with a lower brick."""

        upper_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            upper_body_name,
        )

        lower_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            lower_body_name,
        )

        if upper_id == -1 or lower_id == -1:
            return False

        upper_jnt = self.model.body_jntadr[upper_id]
        lower_jnt = self.model.body_jntadr[lower_id]

        if upper_jnt < 0 or lower_jnt < 0:
            return False

        upper_qadr = self.model.jnt_qposadr[upper_jnt]
        lower_qadr = self.model.jnt_qposadr[lower_jnt]

        upper_pos = self.data.qpos[upper_qadr:upper_qadr + 3].copy()
        lower_pos = self.data.qpos[lower_qadr:lower_qadr + 3].copy()

        upper_quat = self.data.qpos[upper_qadr + 3:upper_qadr + 7].copy()
        lower_quat = self.data.qpos[lower_qadr + 3:lower_qadr + 7].copy()

        upper_half = get_world_half_size(upper_body_name, upper_quat)
        lower_half = get_world_half_size(lower_body_name, lower_quat)

        if upper_half is None or lower_half is None:
            return False

        upper_hx, upper_hy, upper_hz = upper_half
        lower_hx, lower_hy, lower_hz = lower_half


        if upper_pos[2] <= lower_pos[2]:
            return False

        upper_bottom_z = upper_pos[2] - upper_hz
        lower_top_z = lower_pos[2] + lower_hz

        vertical_gap = abs(upper_bottom_z - lower_top_z)

        if vertical_gap > VERTICAL_CONTACT_TOL:
            return False

        overlap = xy_overlap_area(
            upper_pos,
            (upper_hx, upper_hy),
            lower_pos,
            (lower_hx, lower_hy),
        )

        upper_area = 4.0 * upper_hx * upper_hy
        lower_area = 4.0 * lower_hx * lower_hy
        min_area = min(upper_area, lower_area)

        if overlap < MIN_OVERLAP_RATIO * min_area:
            return False

        return True

    def snap_brick_to_base_plate(self, brick_name: str) -> bool:
        """Snap a brick to the nearest valid stud pose on the assembly plate."""

        pair = (ASSEMBLY_BASE_NAME, brick_name)

        if pair in self.welded_pairs:
            return False

        target = self.manifest_target_for_body(brick_name)
        if (target is not None and
                float(target["position"][2]) > BRICK_ON_BASE_CENTER_Z + 0.005):
            # Upper layers must connect to their declared supporting brick,
            # never directly to the plate when the gripper passes near it.
            return False

        brick_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            brick_name,
        )

        if brick_id == -1:
            return False

        brick_jnt = self.model.body_jntadr[brick_id]

        if brick_jnt < 0:
            return False

        qadr = self.model.jnt_qposadr[brick_jnt]
        dofadr = self.model.jnt_dofadr[brick_jnt]

        pos = self.data.qpos[qadr:qadr + 3].copy()
        quat = self.data.qpos[qadr + 3:qadr + 7].copy()


        dx = abs(pos[0] - ASSEMBLY_BASE_CENTER_X)
        dy = abs(pos[1] - ASSEMBLY_BASE_CENTER_Y)

        if dx > BASE_PLATE_HALF_X or dy > BASE_PLATE_HALF_Y:
            return False


        brick_bottom_z = pos[2] - BRICK_BODY_HALF_HEIGHT
        vertical_gap = abs(brick_bottom_z - BASE_PLATE_TOP_Z)

        if vertical_gap > BASE_SNAP_VERTICAL_TOL:
            return False


        snapped_x, snapped_y = snap_xy_to_stud_grid(pos[0], pos[1])
        snapped_z = BRICK_ON_BASE_CENTER_Z
        yaw = yaw_from_quat_wxyz(quat)
        snapped_yaw = snap_yaw_to_90(yaw)
        if target is not None:
            target_xy = np.asarray(target["position"][:2], dtype=float)
            if float(np.linalg.norm(pos[:2] - target_xy)) <= 0.08:
                snapped_x, snapped_y = target_xy
                snapped_z = float(target["position"][2])
                snapped_yaw = float(target.get("yaw_rad", snapped_yaw))
        snapped_quat = quat_wxyz_from_yaw(snapped_yaw)


        self.data.qpos[qadr + 0] = snapped_x
        self.data.qpos[qadr + 1] = snapped_y
        self.data.qpos[qadr + 2] = snapped_z
        self.data.qpos[qadr + 3:qadr + 7] = snapped_quat


        self.data.qvel[dofadr:dofadr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)


        self.fake_welds.append(
            {
                "parent": ASSEMBLY_BASE_NAME,
                "child": brick_name,
                "rel_pos": np.array(
                    [
                        snapped_x - ASSEMBLY_BASE_CENTER_X,
                        snapped_y - ASSEMBLY_BASE_CENTER_Y,
                        snapped_z - TABLE_TOP_Z,
                    ],
                    dtype=float,
                ),
                "child_quat": snapped_quat.copy(),
            }
        )

        self.welded_pairs.add(pair)

        self.get_logger().info(
            f"[BASE_SNAP] {brick_name} -> "
            f"x={snapped_x:.4f}, y={snapped_y:.4f}, "
            f"z={snapped_z:.4f}, yaw={snapped_yaw:.3f}"
        )

        return True

    def manifest_target_for_body(self, body_name: str):
        """Return the declared target corresponding to a MuJoCo body name."""
        if not self.episode_manifest:
            return None
        for target in self.episode_manifest["target_blocks"]:
            if target["body_name"] == body_name:
                return target
        return None

    def snap_brick_to_manifest_target(
            self, body_name: str, max_distance: float = 0.08,
            max_planar_distance: float | None = None) -> bool:
        """Align a near-target block exactly in the documented snap mode."""
        target = self.manifest_target_for_body(body_name)
        if target is None:
            return False
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return False
        joint_id = self.model.body_jntadr[body_id]
        if joint_id < 0:
            return False
        qadr = self.model.jnt_qposadr[joint_id]
        dofadr = self.model.jnt_dofadr[joint_id]
        position = self.data.qpos[qadr:qadr + 3]
        target_position = np.asarray(target["position"], dtype=float)
        correction = float(np.linalg.norm(position - target_position))
        planar_correction = float(np.linalg.norm(position[:2] - target_position[:2]))
        if (correction > max_distance or
                (max_planar_distance is not None and
                 planar_correction > max_planar_distance)):
            return False
        self.data.qpos[qadr:qadr + 3] = target_position
        self.data.qpos[qadr + 3:qadr + 7] = quat_wxyz_from_yaw(
            float(target.get("yaw_rad", 0.0))
        )
        self.data.qvel[dofadr:dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"[TARGET_SNAP] {body_name}, pose_correction={correction:.4f}m"
        )
        return True
    # =========================================================

    # =========================================================

    def auto_weld_touching_bricks(self):
        """Create a fixed relative transform when two compatible brick faces meet."""

        with self.mj_lock:
            # Snapping calls mj_forward(), which can rebuild the contact array.
            # Snapshot geom pairs first so iteration never references stale contacts.
            contact_pairs = [
                (int(self.data.contact[i].geom1), int(self.data.contact[i].geom2))
                for i in range(self.data.ncon)
            ]
            for geom1, geom2 in contact_pairs:
                body1_id = self.model.geom_bodyid[geom1]
                body2_id = self.model.geom_bodyid[geom2]

                body1_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body1_id,
                )

                body2_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body2_id,
                )

                if body1_name is None or body2_name is None:
                    continue


                if body1_name == ASSEMBLY_BASE_NAME and "brick" in body2_name:
                    if body2_name == self.grasped_block:
                        continue
                    self.snap_brick_to_base_plate(body2_name)
                    continue

                if body2_name == ASSEMBLY_BASE_NAME and "brick" in body1_name:
                    if body1_name == self.grasped_block:
                        continue
                    self.snap_brick_to_base_plate(body1_name)
                    continue

                if "brick" not in body1_name or "brick" not in body2_name:
                    continue

                if body1_name == body2_name:
                    continue

                pair = tuple(sorted([body1_name, body2_name]))

                if pair in self.welded_pairs:
                    continue

                id1 = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body1_name,
                )

                id2 = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body2_name,
                )

                if id1 == -1 or id2 == -1:
                    continue

                z1 = self.data.xpos[id1][2]
                z2 = self.data.xpos[id2][2]

                if z1 > z2:
                    upper_name = body1_name
                    lower_name = body2_name
                else:
                    upper_name = body2_name
                    lower_name = body1_name

                if upper_name == self.grasped_block:
                    continue

                if not self.should_weld_bottom_to_top(upper_name, lower_name):
                    continue

                self.snap_brick_to_manifest_target(upper_name)
                self.create_fake_weld(lower_name, upper_name)
                self.welded_pairs.add(pair)

                self.get_logger().info(
                    f"[AUTO_WELD] lower={lower_name}, upper={upper_name}, "
                    f"condition=bottom_to_top_overlap"
                )

    def create_fake_weld(self, parent_name: str, child_name: str):
        parent_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_name,
        )

        child_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_name,
        )

        if parent_id == -1 or child_id == -1:
            return

        child_jnt = self.model.body_jntadr[child_id]

        if child_jnt < 0:
            return

        qadr = self.model.jnt_qposadr[child_jnt]

        parent_rotation = self.data.xmat[parent_id].reshape(3, 3).copy()
        rel_pos = parent_rotation.T @ (
            self.data.xpos[child_id].copy() - self.data.xpos[parent_id].copy()
        )
        parent_quat = self.data.xquat[parent_id].copy()
        child_quat = self.data.qpos[qadr + 3:qadr + 7].copy()
        rel_quat = quat_multiply_wxyz(quat_conjugate_wxyz(parent_quat), child_quat)

        self.fake_welds.append(
            {
                "parent": parent_name,
                "child": child_name,
                "rel_pos": rel_pos,
                "rel_quat": rel_quat,
            }
        )

    def settle_released_block(self, body_name: str) -> bool:
        """Transfer a released block from the hand to its intended support."""
        target = self.manifest_target_for_body(body_name)
        if target is None:
            return False
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        joint_id = self.model.body_jntadr[body_id] if body_id >= 0 else -1
        if joint_id < 0:
            return False
        qadr = self.model.jnt_qposadr[joint_id]
        position = self.data.qpos[qadr:qadr + 3].copy()
        target_position = np.asarray(target["position"], dtype=float)
        delta = position - target_position
        self.get_logger().info(
            f"[RELEASE_HANDOFF] {body_name}, "
            f"delta=({delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f})m"
        )
        if not self.snap_brick_to_manifest_target(
                body_name, max_distance=0.20, max_planar_distance=0.08):
            return False

        if float(target["position"][2]) <= BRICK_ON_BASE_CENTER_Z + 0.005:
            return self.snap_brick_to_base_plate(body_name)

        candidates = []
        for lower in self.episode_manifest["target_blocks"]:
            lower_name = lower["body_name"]
            if lower_name == body_name:
                continue
            if not any(weld["child"] == lower_name for weld in self.fake_welds):
                continue
            if self.should_weld_bottom_to_top(body_name, lower_name):
                candidates.append((float(lower["position"][2]), lower_name))

        if not candidates:
            return False

        _, lower_name = max(candidates)
        pair = tuple(sorted([body_name, lower_name]))
        self.create_fake_weld(lower_name, body_name)
        self.welded_pairs.add(pair)
        self.get_logger().info(
            f"[RELEASE_SNAP] lower={lower_name}, upper={body_name}"
        )
        return True

    def finger_contacts_for_block(self, body_name: str) -> tuple[bool, bool]:
        """Report whether both physical fingertips contact a block."""
        block_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        left_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        right_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        if min(block_id, left_id, right_id) < 0:
            return False, False

        touching = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 == block_id:
                touching.add(body2)
            elif body2 == block_id:
                touching.add(body1)
        return left_id in touching, right_id in touching

    def update_grasp_constraint(self):
        """Track a contact-verified grasp and optionally attach it in snap mode."""
        if not self.episode_manifest:
            return
        with self.mj_lock:
            if self.gripper_mode in {"open", "opening"}:
                self.grasp_contact_body = None
                self.grasp_contact_since = None
                if self.grasped_block is not None:
                    released = self.grasped_block
                    self.update_grasp_slip_locked()
                    self.grasp_translation_slip_by_body[released] = (
                        self.active_grasp_max_translation_slip
                    )
                    self.grasp_rotation_slip_by_body[released] = (
                        self.active_grasp_max_rotation_slip_deg
                    )
                    if self.connection_mode == "snap":
                        self.fake_welds = [
                            weld for weld in self.fake_welds
                            if not (weld["parent"] == "hand" and weld["child"] == released)
                        ]
                        self.welded_pairs.discard(("hand", released))
                    self.grasped_block = None
                    self.get_logger().info(
                        f"[GRASP_RELEASE] {released}, "
                        f"max_airborne_translation_slip="
                        f"{self.active_grasp_max_translation_slip:.4f}m, "
                        f"delta_at_max=("
                        f"{self.active_grasp_translation_delta_at_max[0]:+.4f}, "
                        f"{self.active_grasp_translation_delta_at_max[1]:+.4f}, "
                        f"{self.active_grasp_translation_delta_at_max[2]:+.4f})m, "
                        f"max_airborne_rotation_slip="
                        f"{self.active_grasp_max_rotation_slip_deg:.2f}deg"
                    )
                    self.grasp_reference_offset = None
                    self.grasp_reference_rotation = None
                    if self.connection_mode == "snap":
                        self.settle_released_block(released)
                return
            if self.grasped_block is not None:
                self.update_grasp_slip_locked()
                return
            if self.gripper_mode not in {"closing", "close_hold"}:
                return

            grasp_center = grasp_center_world(self.model, self.data)
            if grasp_center is None:
                return
            hand_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
            )
            hand_rotation = self.data.xmat[hand_id].reshape(3, 3)
            tool_tilt_deg = float(np.degrees(np.arccos(np.clip(
                -hand_rotation[2, 2], -1.0, 1.0
            ))))
            candidates = []
            observed = []
            for item in self.episode_manifest["spawned_blocks"]:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
                )
                if body_id < 0:
                    continue
                delta = block_collision_center_world(self.data, body_id) - grasp_center
                planar = float(np.linalg.norm(delta[:2]))
                vertical = abs(float(delta[2]))
                distance = float(np.linalg.norm(delta))
                left_contact, right_contact = self.finger_contacts_for_block(
                    item["body_name"]
                )
                upright_z = float(self.data.xmat[body_id][8])
                observed.append((
                    distance, item["body_name"], planar, vertical,
                    left_contact, right_contact,
                ))
                if (left_contact and right_contact and distance <= 0.020 and
                        planar <= 0.012 and vertical <= 0.018 and upright_z >= 0.7):
                    if tool_tilt_deg <= GRASP_MAX_TOOL_TILT_DEG:
                        candidates.append((distance, item["body_name"]))
            if not candidates:
                self.grasp_contact_body = None
                self.grasp_contact_since = None
                now = time.monotonic()
                if (self.gripper_mode == "close_hold" and observed and
                        now - self._last_grasp_wait_log >= 1.0):
                    distance, body_name, planar, vertical, left, right = min(observed)
                    finger_positions = [
                        float(self.data.qpos[self.joint_qpos_addr[name]])
                        for name in self.finger_joint_names
                    ]
                    self.get_logger().info(
                        f"[GRASP_WAIT] nearest={body_name}, "
                        f"left_contact={str(left).lower()}, "
                        f"right_contact={str(right).lower()}, "
                        f"distance={distance:.4f}m, planar={planar:.4f}m, "
                        f"vertical={vertical:.4f}m, "
                        f"tool_tilt={tool_tilt_deg:.2f}deg, "
                        f"fingers=({finger_positions[0]:.4f}, "
                        f"{finger_positions[1]:.4f})m"
                    )
                    self._last_grasp_wait_log = now
                return
            _, body_name = min(candidates)
            now = time.monotonic()
            if self.grasp_contact_body != body_name:
                self.grasp_contact_body = body_name
                self.grasp_contact_since = now
                return
            if (self.grasp_contact_since is None or
                    now - self.grasp_contact_since < GRASP_CONTACT_HOLD_S):
                return
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            correction = float(np.linalg.norm(
                block_collision_center_world(self.data, body_id) - grasp_center
            ))
            self.grasped_block = body_name
            block_rotation = self.data.xmat[body_id].reshape(3, 3)
            self.grasp_reference_offset = hand_rotation.T @ (
                block_collision_center_world(self.data, body_id) - grasp_center
            )
            self.grasp_reference_rotation = hand_rotation.T @ block_rotation
            self.active_grasp_max_translation_slip = 0.0
            self.active_grasp_translation_delta_at_max = np.zeros(3, dtype=float)
            self.active_grasp_max_rotation_slip_deg = 0.0
            self.active_grasp_last_logged_slip = 0.0
            self.grasp_airborne_started = False
            self.grasp_airborne_finished = False
            self.grasp_airborne_since = None
            contact_time = now - self.grasp_contact_since
            if self.connection_mode == "snap":
                self.create_fake_weld("hand", body_name)
                self.welded_pairs.add(("hand", body_name))
                event = "GRASP_ATTACH"
            else:
                event = "GRASP_PHYSICS_CONFIRMED"
            self.get_logger().info(
                f"[{event}] {body_name}, dual_finger_contact=true, "
                f"contact_time={contact_time:.3f}s, center_error={correction:.4f}m, "
                f"tool_tilt={tool_tilt_deg:.2f}deg, "
                f"center_delta=({self.grasp_reference_offset[0]:+.4f}, "
                f"{self.grasp_reference_offset[1]:+.4f}, "
                f"{self.grasp_reference_offset[2]:+.4f})m"
            )

    def update_grasp_slip_locked(self):
        """Measure part motion relative to the hand while a physical grasp is active."""
        if (self.grasped_block is None or self.grasp_reference_offset is None or
                self.grasp_reference_rotation is None):
            return
        hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.grasped_block
        )
        if min(hand_id, body_id) < 0:
            return
        grasp_center = grasp_center_world(self.model, self.data)
        if grasp_center is None:
            return
        hand_rotation = self.data.xmat[hand_id].reshape(3, 3)
        block_rotation = self.data.xmat[body_id].reshape(3, 3)
        current_offset = hand_rotation.T @ (
            block_collision_center_world(self.data, body_id) - grasp_center
        )
        finger_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("hand", "left_finger", "right_finger")
        }
        support_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 == body_id and body2 not in finger_ids:
                support_contact = True
                break
            if body2 == body_id and body1 not in finger_ids:
                support_contact = True
                break
        if support_contact:
            if self.grasp_airborne_started:
                self.grasp_airborne_finished = True
            else:
                self.grasp_airborne_since = None
            return
        if self.grasp_airborne_finished:
            return
        if not self.grasp_airborne_started:
            if self.grasp_airborne_since is None:
                self.grasp_airborne_since = float(self.data.time)
                return
            if float(self.data.time) - self.grasp_airborne_since < GRASP_AIRBORNE_SETTLE_S:
                return
            self.grasp_reference_offset = current_offset.copy()
            self.grasp_reference_rotation = hand_rotation.T @ block_rotation
            self.grasp_airborne_started = True
            return
        translation_delta = current_offset - self.grasp_reference_offset
        translation_slip = float(np.linalg.norm(translation_delta))
        current_rotation = hand_rotation.T @ block_rotation
        rotation_delta = self.grasp_reference_rotation.T @ current_rotation
        cosine = float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_slip_deg = float(np.degrees(np.arccos(cosine)))
        if translation_slip > self.active_grasp_max_translation_slip:
            self.active_grasp_max_translation_slip = translation_slip
            self.active_grasp_translation_delta_at_max = translation_delta.copy()
            if translation_slip - self.active_grasp_last_logged_slip >= 0.0005:
                self.get_logger().info(
                    f"[GRASP_SLIP_MAX] {self.grasped_block}, "
                    f"translation={translation_slip:.4f}m, "
                    f"delta=({translation_delta[0]:+.4f}, "
                    f"{translation_delta[1]:+.4f}, "
                    f"{translation_delta[2]:+.4f})m"
                )
                self.active_grasp_last_logged_slip = translation_slip
        self.active_grasp_max_rotation_slip_deg = max(
            self.active_grasp_max_rotation_slip_deg, rotation_slip_deg
        )

    def maintain_fake_welds(self):
        """Maintain relative transforms created by the simplified snap model."""
        with self.mj_lock:
            for weld in self.fake_welds:
                parent_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    weld["parent"],
                )

                child_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    weld["child"],
                )

                if parent_id == -1 or child_id == -1:
                    continue

                child_jnt = self.model.body_jntadr[child_id]

                if child_jnt < 0:
                    continue

                qadr = self.model.jnt_qposadr[child_jnt]
                dofadr = self.model.jnt_dofadr[child_jnt]

                if weld["parent"] == ASSEMBLY_BASE_NAME:
                    target_pos = np.array(
                        [
                            ASSEMBLY_BASE_CENTER_X + weld["rel_pos"][0],
                            ASSEMBLY_BASE_CENTER_Y + weld["rel_pos"][1],
                            BRICK_ON_BASE_CENTER_Z,
                        ],
                        dtype=float,
                    )
                    target_quat = weld["child_quat"]
                else:
                    parent_rotation = self.data.xmat[parent_id].reshape(3, 3)
                    target_pos = self.data.xpos[parent_id] + parent_rotation @ weld["rel_pos"]
                    target_quat = quat_multiply_wxyz(
                        self.data.xquat[parent_id], weld["rel_quat"]
                    )

                self.data.qpos[qadr + 0] = target_pos[0]
                self.data.qpos[qadr + 1] = target_pos[1]
                self.data.qpos[qadr + 2] = target_pos[2]
                self.data.qpos[qadr + 3:qadr + 7] = target_quat

                self.data.qvel[dofadr:dofadr + 6] = 0.0

            if self.fake_welds:
                mujoco.mj_forward(self.model, self.data)

    # =========================================================

    # =========================================================

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        with self.mj_lock:
            for name in self.all_joint_names:
                if name not in self.joint_qpos_addr:
                    continue

                qadr = self.joint_qpos_addr[name]
                vadr = self.joint_qvel_addr[name]

                msg.name.append(name)
                msg.position.append(float(self.data.qpos[qadr]))
                msg.velocity.append(float(self.data.qvel[vadr]))
                msg.effort.append(0.0)

        self.joint_state_pub.publish(msg)


def main():
    rclpy.init()

    node = None
    executor = None

    try:
        node = MuJoCoActionServer()

        executor = MultiThreadedExecutor()
        executor.add_node(node)

        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()

        def run_loop(viewer=None):
            node.simulation_loop_ready.set()
            node.get_logger().info("Simulation and controller loop is ready")
            try:
                while rclpy.ok() and (viewer is None or viewer.is_running()):
                    loop_start = time.time()


                    node.update_action_state()


                    node.update_gripper_target()

                    for _ in range(SIM_SUBSTEPS):
                        node.step_pid()

                        node.update_grasp_constraint()


                        if node.connection_mode == "snap":
                            node.auto_weld_touching_bricks()
                            node.maintain_fake_welds()

                    node.update_safety_metrics()

                    # MuJoCo's OpenGL context is thread-affine, so rendering
                    # stays on this thread. The visual executor requests one
                    # fresh frame only after reaching its stationary observation
                    # pose. Other RGB-D clients retain the regular stream by
                    # leaving on-demand mode disabled.
                    if node.camera_renderer is not None:
                        capture_id = node.pending_camera_capture()
                        if capture_id is not None:
                            node.publish_virtual_camera(capture_id)
                            node._last_camera_publish = time.monotonic()
                        elif (not node.camera_on_demand and
                                time.monotonic() - node._last_camera_publish >= 0.5):
                            node.publish_virtual_camera()
                            node._last_camera_publish = time.monotonic()

                    if viewer is not None:
                        viewer.sync()

                    elapsed = time.time() - loop_start
                    sleep_time = LOOP_DT - elapsed

                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                node.simulation_loop_ready.clear()

        if os.environ.get("MJ_BRIDGE_HEADLESS", "0").lower() in ("1", "true", "yes"):
            node.get_logger().info("Running in headless mode")
            run_loop()
        else:
            with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
                run_loop(viewer)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if executor is not None:
                executor.shutdown()

            if node is not None:
                if node.camera_renderer is not None:
                    node.camera_renderer.close()
                node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
