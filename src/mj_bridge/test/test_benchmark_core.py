import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import yaml

from mj_bridge.benchmark_core import generate_episode, normalize_product, score_episode
from mj_bridge.benchmark_cli import generate as generate_files
from mj_bridge.executor_planner import plan_assembly
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
