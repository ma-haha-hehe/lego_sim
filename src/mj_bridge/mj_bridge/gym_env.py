"""Direct MuJoCo environment for Python policies that do not need ROS 2."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from .benchmark_cli import generate
from .benchmark_core import dump_json, load_yaml, score_episode


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LegoBenchEnv:
    """Gymnasium-style API backed directly by the generated MuJoCo model.

    Actions are absolute MuJoCo actuator controls in model actuator order. The
    environment intentionally supports only raw contact physics; use the ROS 2
    bridge when deterministic snap-mode connections are required.
    """

    metadata = {"render_modes": ["rgb_array", "depth_array"]}

    def __init__(
        self,
        product: str | Path,
        *,
        seed: int = 0,
        output_dir: str | Path = "runs/python-env",
        observation_mode: str = "oracle",
        render_width: int = 320,
        render_height: int = 240,
        frame_skip: int = 20,
        max_episode_steps: int = 5000,
    ) -> None:
        if observation_mode not in {"oracle", "rgbd"}:
            raise ValueError("observation_mode must be 'oracle' or 'rgbd'")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.product_path = Path(product).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.observation_mode = observation_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.seed = int(seed)

        self.episode_manifest: dict[str, Any] = {}
        self.execution_plan: dict[str, Any] = {}
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self.renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._initial_qpos = np.empty(0, dtype=float)
        self._initial_qvel = np.empty(0, dtype=float)
        self._initial_ctrl = np.empty(0, dtype=float)
        self._robot_joint_ids: list[int] = []
        self._arm_dof_addresses: list[int] = []
        self._load_episode()

    def _load_episode(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        self.episode_manifest, scene_path = generate(
            str(self.product_path), self.seed, str(self.output_dir)
        )
        self.execution_plan = load_yaml(self.output_dir / "execution_plan.yaml")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self._robot_joint_ids = []
        self._arm_dof_addresses = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and name.startswith("panda_"):
                self._robot_joint_ids.append(joint_id)
                if name.startswith("panda_joint"):
                    self._arm_dof_addresses.append(int(self.model.jnt_dofadr[joint_id]))
        self._configure_initial_state()
        if self.observation_mode == "rgbd":
            self.renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        self._reset_model()

    def _configure_initial_state(self) -> None:
        configuration = load_yaml(Path(__file__).with_name("initial_positions.yaml"))
        positions = configuration.get("initial_positions", {})
        for name, value in positions.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = float(value)
        for name in ("panda_finger_joint1", "panda_finger_joint2"):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = 0.04
        mujoco.mj_forward(self.model, self.data)

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_address]
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._initial_ctrl = self.data.ctrl.copy()

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            or f"actuator_{index}"
            for index in range(self.model.nu)
        )

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.model.nu,)

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        limited = self.model.actuator_ctrllimited.astype(bool)
        lower = np.where(limited, self.model.actuator_ctrlrange[:, 0], -np.inf)
        upper = np.where(limited, self.model.actuator_ctrlrange[:, 1], np.inf)
        return lower.copy(), upper.copy()

    @property
    def neutral_action(self) -> np.ndarray:
        """Return the reset-time actuator command."""
        return self._initial_ctrl.copy()

    def _reset_model(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        if self.model.nu:
            self.data.ctrl[:] = self._initial_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the same episode, or regenerate it when the seed changes."""
        del options
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            self._load_episode()
        else:
            self._reset_model()
        score = self.score()
        return self.observation(), self._info(score)

    def _robot_state(self) -> dict[str, Any]:
        names, positions, velocities = [], [], []
        for joint_id in self._robot_joint_ids:
            names.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            )
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            positions.append(float(self.data.qpos[qpos_address]))
            velocities.append(float(self.data.qvel[dof_address]))
        return {
            "joint_names": names,
            "joint_positions": np.asarray(positions, dtype=np.float64),
            "joint_velocities": np.asarray(velocities, dtype=np.float64),
            "actuator_controls": self.data.ctrl.copy(),
        }

    def _authoritative_state(self) -> dict[str, Any]:
        blocks = {}
        for item in self.episode_manifest["spawned_blocks"]:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
            )
            quaternion = self.data.xquat[body_id].copy()
            blocks[item["id"]] = {
                "type": item["type"],
                "color": item["color"],
                "body_name": item["body_name"],
                "position": self.data.xpos[body_id].astype(float).tolist(),
                "quaternion_wxyz": quaternion.astype(float).tolist(),
                "yaw_rad": _yaw_from_wxyz(quaternion),
            }
        return {
            "episode_id": self.episode_manifest["episode_id"],
            "blocks": blocks,
        }

    def observation(self) -> dict[str, Any]:
        """Return policy-visible state without RGB-D ground-truth leakage."""
        observation: dict[str, Any] = {
            "episode_id": self.episode_manifest["episode_id"],
            "seed": self.seed,
            "simulation_time_s": float(self.data.time),
            "targets": self.episode_manifest["target_blocks"],
            "robot": self._robot_state(),
        }
        if self.observation_mode == "oracle":
            observation["blocks"] = self._authoritative_state()["blocks"]
        else:
            observation.update(self.render_rgbd())
        return observation

    def render_rgbd(self) -> dict[str, np.ndarray]:
        if self.renderer is None:
            raise RuntimeError("RGB-D rendering requires observation_mode='rgbd'")
        self.renderer.update_scene(self.data, camera="realsense")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="realsense")
        depth = self.renderer.render().astype(np.float32).copy()
        self.renderer.disable_depth_rendering()
        return {"rgb": rgb, "depth": depth}

    def score(self) -> dict[str, Any]:
        return score_episode(self.episode_manifest, self._authoritative_state())

    def _info(self, score: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": score,
            "step_count": self._step_count,
            "connection_mode": "physics",
            "execution_plan": self.execution_plan,
        }

    def step(
        self,
        action: Sequence[float] | np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one absolute actuator command and advance the simulation."""
        command = np.asarray(action, dtype=float)
        if command.shape != self.action_shape:
            raise ValueError(
                f"action must have shape {self.action_shape}, received {command.shape}"
            )
        if not np.all(np.isfinite(command)):
            raise ValueError("action must contain only finite values")
        lower, upper = self.action_bounds
        self.data.ctrl[:] = np.clip(command, lower, upper)
        for _ in range(self.frame_skip):
            for dof_address in self._arm_dof_addresses:
                self.data.qfrc_applied[dof_address] = self.data.qfrc_bias[dof_address]
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        score = self.score()
        terminated = bool(score["success"])
        truncated = self._step_count >= self.max_episode_steps and not terminated
        reward = float(score["completion"])
        return self.observation(), reward, terminated, truncated, self._info(score)

    def export_result(self) -> dict[str, Any]:
        """Write actual_state.json and result.json into the episode directory."""
        actual = self._authoritative_state()
        result = score_episode(self.episode_manifest, actual)
        result.update({
            "episode_id": self.episode_manifest["episode_id"],
            "seed": self.seed,
            "simulation_time_s": float(self.data.time),
            "connection_mode": "physics",
            "observation_mode": self.observation_mode,
        })
        dump_json(self.output_dir / "actual_state.json", actual)
        dump_json(self.output_dir / "result.json", result)
        return result

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> "LegoBenchEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
