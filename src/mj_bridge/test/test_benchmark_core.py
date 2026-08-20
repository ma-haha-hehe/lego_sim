import math

from mj_bridge.benchmark_core import generate_episode, normalize_product, score_episode


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
