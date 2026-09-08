import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from mj_bridge.benchmark_core import generate_episode, normalize_product, score_episode
from mj_bridge.benchmark_cli import generate as generate_files
from mj_bridge.executor_planner import plan_assembly
from mj_bridge.gym_env import LegoBenchEnv
from mj_bridge.grasp_geometry import (
    align_block_to_grasp_center, block_collision_center_world, grasp_center_world,
)


def sample_product():
    return normalize_product({
        "blocks": [
            {"name": "red_base", "type": "brick_2x2", "color": "red", "pos": [0, 0, 0]},
            {"name": "blue_top", "type": "brick_4x2", "color": "blue", "pos": [0, 0, 0.0192]},
        ]
    })


def test_seed_is_reproducible():
    a = generate_episode(sample_product(), seed=17)
    b = generate_episode(sample_product(), seed=17)
    assert a["spawned_blocks"] == b["spawned_blocks"]


def test_spawned_parts_use_orthogonal_yaw_only():
    observed = set()
    for seed in range(12):
        episode = generate_episode(sample_product(), seed=seed)
        assert episode["spawn_yaw_choices_deg"] == [0.0, 90.0]
        for block in episode["spawned_blocks"]:
            observed.add(round(float(block["yaw_rad"]), 8))
    assert observed == {0.0, round(math.pi / 2.0, 8)}


def test_spawned_parts_do_not_overlap_conservative_aabbs():
    episode = generate_episode(sample_product(), seed=9)
    a, b = episode["spawned_blocks"]
    assert math.dist(a["position"][:2], b["position"][:2]) > 0.04


def test_perfect_target_state_scores_success():
    episode = generate_episode(sample_product(), seed=2)
    actual = {"blocks": {
        b["id"]: {"position": b["position"], "yaw_rad": b["yaw_rad"]}
        for b in episode["target_blocks"]
    }}
    result = score_episode(episode, actual)
    assert result["success"] is True
    assert result["completion"] == 1.0


def test_visually_identical_parts_are_scored_as_interchangeable():
    product = normalize_product({"blocks": [
        {"name": "left", "type": "brick_2x2", "color": "blue", "pos": [-0.016, 0, 0]},
        {"name": "right", "type": "brick_2x2", "color": "blue", "pos": [0.016, 0, 0]},
    ]})
    episode = generate_episode(product, seed=3)
    left, right = episode["target_blocks"]
    actual = {"blocks": {
        "left": {"type": "brick_2x2", "color": "blue",
                 "position": right["position"], "yaw_rad": right["yaw_rad"]},
        "right": {"type": "brick_2x2", "color": "blue",
                  "position": left["position"], "yaw_rad": left["yaw_rad"]},
    }}
    result = score_episode(episode, actual)
    assert result["success"] is True
    assert result["placed"] == 2
    assert {item.get("observed_id") for item in result["blocks"]} == {"left", "right"}


def test_legacy_radians_are_normalized_to_degrees():
    product = normalize_product({"blocks": [{
        "name": "2x2_test", "type": "brick_2x2", "pos": [0, 0, 0],
        "rotation": [0, 0, math.pi / 2],
    }]})
    assert abs(product["blocks"][0]["target"]["yaw_deg"] - 90.0) < 1e-6


def test_legacy_duplicate_names_get_stable_unique_ids():
    product = normalize_product({"blocks": [
        {"name": "2x2_same", "pos": [0, 0, 0]},
        {"name": "2x2_same", "pos": [0, 0, 0.02]},
    ]})
    assert [block["id"] for block in product["blocks"]] == ["2x2_same", "2x2_same_2"]


def test_one_degree_is_not_misread_as_one_radian():
    product = normalize_product({"blocks": [{
        "name": "2x2_test", "pos": [0, 0, 0], "rotation": [0, 0, 1],
    }]})
    assert product["blocks"][0]["target"]["yaw_deg"] == 1.0


def test_planner_orders_supports_before_upper_blocks():
    product = normalize_product({"blocks": [
        {"name": "top", "type": "brick_2x2", "pos": [0, 0, 0.0192]},
        {"name": "base", "type": "brick_2x2", "pos": [0, 0, 0]},
    ]})
    plan = plan_assembly(product)
    assert [step["id"] for step in plan["steps"]] == ["base", "top"]
    assert plan["steps"][1]["depends_on"] == ["base"]


def test_planner_keeps_stable_ids_for_identical_parts():
    plan = plan_assembly(sample_product())
    assert [step["id"] for step in plan["steps"]] == ["red_base", "blue_top"]


def test_planner_records_multiple_direct_supports():
    product = normalize_product({"schema_version": 1, "product": {"name": "bridge"}, "blocks": [
        {"id": "left", "type": "brick_2x2", "target": {"position": [-0.016, 0, 0]}},
        {"id": "right", "type": "brick_2x2", "target": {"position": [0.016, 0, 0]}},
        {"id": "beam", "type": "brick_4x2", "target": {"position": [0, 0, 0.0192]}},
    ]})
    plan = plan_assembly(product)
    assert plan["steps"][2]["id"] == "beam"
    assert plan["steps"][2]["depends_on"] == ["left", "right"]


def test_planner_prefers_90_degree_grasp_for_clear_2x4():
    product = normalize_product({"blocks": [
        {"name": "beam", "type": "brick_4x2", "pos": [0, 0, 0]},
    ]})
    plan = plan_assembly(product)
    assert plan["planning_strategy"] == "reverse_disassembly_90_first"
    assert plan["grasp_angle_frame"] == "part_local"
    assert plan["steps"][0]["grasp_spin_deg"] == 90.0


def test_planner_grasp_angle_is_relative_to_rotated_2x4():
    product = normalize_product({"blocks": [
        {
            "name": "beam",
            "type": "brick_4x2",
            "pos": [0, 0, 0],
            "rotation": [0, 0, 90],
        },
        {"name": "local_x_obstacle", "type": "brick_2x2", "pos": [0, 0.032, 0]},
    ]})
    plan = plan_assembly(product)
    beam_step = next(step for step in plan["steps"] if step["id"] == "beam")
    assert beam_step["grasp_spin_deg"] == 0.0


def test_planner_reverses_same_level_disassembly_order():
    product = normalize_product({"blocks": [
        {"name": "first", "type": "brick_2x2", "pos": [-0.08, 0, 0]},
        {"name": "second", "type": "brick_2x2", "pos": [0.08, 0, 0]},
    ]})
    plan = plan_assembly(product)
    assert [step["id"] for step in plan["steps"]] == ["second", "first"]


def test_table_contact_resists_robot_scale_downward_force():
    product = normalize_product({"blocks": [
        {"name": "test_block", "type": "brick_2x2", "pos": [0, 0, 0]},
    ]})
    episode = generate_episode(product, seed=5)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        episode, scene = generate_files(str(product_file), 5, str(directory / "run"))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, episode["spawned_blocks"][0]["body_name"]
        )
        minimum_collision_bottom = float("inf")
        for step in range(2000):
            data.xfrc_applied[body_id, 2] = -10.0 * min(1.0, step / 1000.0)
            mujoco.mj_step(model, data)
            collision_bottom = float(data.xpos[body_id, 2]) - 0.009 - 0.0095
            minimum_collision_bottom = min(minimum_collision_bottom, collision_bottom)
        assert minimum_collision_bottom >= 0.0395


def test_generated_brick_uses_registered_mass_and_box_inertia():
    product = normalize_product({"blocks": [
        {"name": "small", "type": "brick_2x2", "pos": [0, 0, 0]},
        {"name": "large", "type": "brick_4x2", "pos": [0, 0, 0.0192]},
    ]})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        episode, scene = generate_files(str(product_file), 12, str(directory / "run"))
        model = mujoco.MjModel.from_xml_path(str(scene))
        expected_masses = {"small": 0.012, "large": 0.022}
        for item in episode["spawned_blocks"]:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
            )
            assert math.isclose(
                float(model.body_mass[body_id]), expected_masses[item["id"]], rel_tol=1e-9
            )


def test_hollow_collision_shell_allows_three_physics_layers_to_settle():
    product = normalize_product({"blocks": [
        {"name": "base", "type": "brick_2x2", "pos": [0, 0, 0]},
        {"name": "middle", "type": "brick_2x2", "pos": [0, 0, 0.0192]},
        {"name": "top", "type": "brick_2x2", "pos": [0, 0, 0.0384]},
    ]})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        episode, scene = generate_files(str(product_file), 42, str(directory / "run"))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for target in episode["target_blocks"]:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, target["body_name"]
            )
            joint_id = int(model.body_jntadr[body_id])
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            yaw = float(target["yaw_rad"])
            data.qpos[qpos_address:qpos_address + 3] = [
                target["position"][0], target["position"][1], target["position"][2] + 0.002,
            ]
            data.qpos[qpos_address + 3:qpos_address + 7] = [
                math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0),
            ]
            data.qvel[dof_address:dof_address + 6] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(1500):
                mujoco.mj_step(model, data)

        for target in episode["target_blocks"]:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, target["body_name"]
            )
            error = data.xpos[body_id] - np.asarray(target["position"], dtype=float)
            assert np.linalg.norm(error[:2]) < 0.001
            assert abs(float(error[2])) < 0.001


def test_direct_python_environment_reset_step_and_export():
    product = normalize_product({"blocks": [
        {"name": "test_block", "type": "brick_2x2", "pos": [0, 0, 0]},
    ]})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        with LegoBenchEnv(
            product_file,
            seed=21,
            output_dir=directory / "episode",
            observation_mode="oracle",
            frame_skip=2,
            max_episode_steps=2,
        ) as environment:
            observation, info = environment.reset()
            assert observation["episode_id"] == "product-seed-21"
            assert "test_block" in observation["blocks"]
            assert len(observation["robot"]["joint_names"]) == 9
            assert environment.neutral_action.shape == environment.action_shape
            assert math.isclose(
                float(observation["robot"]["joint_positions"][1]), -0.785,
                abs_tol=1e-9,
            )
            assert np.allclose(
                observation["robot"]["joint_positions"], environment.neutral_action
            )
            assert info["connection_mode"] == "physics"

            with pytest.raises(ValueError, match="action must have shape"):
                environment.step(np.zeros(environment.model.nu + 1))

            observation, reward, terminated, truncated, info = environment.step(
                environment.neutral_action
            )
            assert observation["simulation_time_s"] > 0.0
            assert np.max(np.abs(
                observation["robot"]["joint_positions"] - environment.neutral_action
            )) < 0.01
            assert 0.0 <= reward <= 1.0
            assert not (terminated and truncated)
            result = environment.export_result()
            assert result["connection_mode"] == "physics"
            assert (directory / "episode" / "actual_state.json").is_file()
            assert (directory / "episode" / "result.json").is_file()

            previous_position = observation["blocks"]["test_block"]["position"]
            regenerated, info = environment.reset(seed=22)
            assert regenerated["episode_id"] == "product-seed-22"
            assert regenerated["blocks"]["test_block"]["position"] != previous_position
            assert info["step_count"] == 0


def test_snap_grasp_aligns_visual_and_collision_center_between_fingers():
    product = normalize_product({"blocks": [
        {"name": "test_block", "type": "brick_2x2", "pos": [0, 0, 0]},
    ]})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        episode, scene = generate_files(str(product_file), 6, str(directory / "run"))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, episode["spawned_blocks"][0]["body_name"]
        )
        correction = align_block_to_grasp_center(model, data, body_id)
        assert np.isfinite(correction)
        assert np.linalg.norm(
            block_collision_center_world(data, body_id) - grasp_center_world(model, data)
        ) < 1e-9


def test_rubber_gripper_pads_have_more_friction_than_abs_parts():
    product = normalize_product({"blocks": [
        {"name": "test_block", "type": "brick_2x2", "pos": [0, 0, 0]},
    ]})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        product_file = directory / "product.yaml"
        product_file.write_text(yaml.safe_dump(product), encoding="utf-8")
        episode, scene = generate_files(str(product_file), 8, str(directory / "run"))
        model = mujoco.MjModel.from_xml_path(str(scene))

        for finger_name in ("left_finger", "right_finger"):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, finger_name
            )
            geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
            pad_ids = [
                geom_id for geom_id in geom_ids
                if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_BOX
            ]
            assert len(pad_ids) == 5
            assert all(int(model.geom_condim[geom_id]) == 6 for geom_id in pad_ids)
            assert all(
                np.all(model.geom_friction[geom_id] >= [12.0, 1.0, 0.10])
                for geom_id in pad_ids
            )

        block_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            episode["spawned_blocks"][0]["body_name"],
        )
        block_geom_ids = np.flatnonzero(model.geom_bodyid == block_id)
        assert block_geom_ids.size > 0
        assert all(
            np.allclose(
                model.geom_friction[geom_id], [1.5, 0.05, 0.005], atol=1e-9
            )
            for geom_id in block_geom_ids
        )
        assert all(
            model.geom_friction[geom_id, 0]
            < model.geom_friction[pad_ids[0], 0]
            for geom_id in block_geom_ids
        )
