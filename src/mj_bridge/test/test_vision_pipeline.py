import numpy as np
import pytest

from mj_bridge.vision_pipeline import (
    FoundationPoseBackend,
    GeometryBackend,
    _camera_point_to_world,
    foundationpose_preflight,
    make_backend,
)


def test_geometry_backend_uses_rgb_and_metric_depth():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[45:75, 65:95] = (0, 160, 0)
    depth = np.full((120, 160), 1.0, dtype=np.float32)
    intrinsics = np.array(
        [[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    detections = GeometryBackend().infer(
        rgb,
        depth,
        intrinsics,
        np.array([0.5, -0.1, 1.1]),
        [{"type": "brick_2x2", "color": "green"}],
    )

    assert len(detections) == 1
    detected = detections[0]
    assert detected.backend == "rgbd_geometry"
    assert detected.part_type == "brick_2x2"
    assert detected.color == "green"
    assert detected.position == pytest.approx([0.496875, -0.096875, 0.0905])


def test_geometry_backend_separates_a_white_part_from_a_white_table():
    rgb = np.full((120, 160, 3), 230, dtype=np.uint8)
    depth = np.full((120, 160), 1.06, dtype=np.float32)
    depth[45:75, 50:110] = 1.041
    intrinsics = np.array(
        [[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    detections = GeometryBackend().infer(
        rgb,
        depth,
        intrinsics,
        np.array([0.5, -0.1, 1.1]),
        [{"type": "brick_4x2", "color": "white"}],
    )

    assert len(detections) == 1
    assert detections[0].part_type == "brick_4x2"
    assert detections[0].color == "white"


def test_oblique_camera_coordinates_are_transformed_to_world():
    camera_position = np.array([0.85, 0.35, 0.55])
    camera_rotation = np.array(
        [
            [0.0, 0.5948430054, -0.8038418992],
            [1.0, 0.0, 0.0],
            [0.0, -0.8038418992, -0.5948430054],
        ]
    )

    world = _camera_point_to_world(
        np.array([0.02, -0.03, 0.40]), camera_position, camera_rotation
    )

    assert world == pytest.approx(camera_position + camera_rotation @ [0.02, -0.03, 0.40])


def test_foundationpose_backend_requires_an_external_install():
    with pytest.raises(RuntimeError, match="foundationpose_root is empty"):
        make_backend("foundationpose", foundationpose_root="")


def test_foundationpose_mesh_uses_the_simulator_body_frame():
    mesh = FoundationPoseBackend._mesh("brick_2x2")

    assert mesh.bounds[0] == pytest.approx([-0.016, -0.016, -0.0095])
    assert mesh.bounds[1] == pytest.approx([0.016, 0.016, 0.0135])


def test_foundationpose_symmetries_match_part_geometry():
    square = FoundationPoseBackend._symmetry_transforms("brick_2x2")
    rectangle = FoundationPoseBackend._symmetry_transforms("brick_4x2")

    assert square.shape == (4, 4, 4)
    assert rectangle.shape == (2, 4, 4)
    np.testing.assert_allclose(
        square[1, :2, :2], [[0.0, -1.0], [1.0, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        rectangle[1, :2, :2], [[-1.0, 0.0], [0.0, -1.0]], atol=1e-6
    )


def test_foundationpose_preflight_reports_missing_checkout(tmp_path):
    report = foundationpose_preflight(str(tmp_path / "missing"))

    assert report["ready"] is False
    assert report["checks"][0]["name"] == "foundationpose_root"
    assert report["checks"][0]["passed"] is False


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown vision backend"):
        make_backend("not-a-backend")
