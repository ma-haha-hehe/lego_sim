#!/usr/bin/env python3
"""
RGB-D perception backends used by the visual benchmark executor.

The production backend follows the same sequence as the hardware workspace:
GroundingDINO proposes boxes, SAM turns them into masks, and FoundationPose
estimates the object pose from the masked RGB-D crop.  A lightweight geometry
backend is also provided for simulator smoke tests on machines without CUDA.
It is deliberately opt-in and is never presented as a FoundationPose result.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PART_HEIGHT_M = {
    "brick_2x2": 0.019,
    "brick_4x2": 0.019,
}

COLOR_RANGES = {
    "red": [((0, 80, 70), (10, 255, 255)), ((170, 80, 70), (180, 255, 255))],
    "green": [((38, 65, 45), (88, 255, 255))],
    "blue": [((92, 65, 45), (138, 255, 255))],
    "yellow": [((18, 70, 70), (42, 255, 255))],
    "orange": [((5, 70, 70), (22, 255, 255))],
    "white": [((0, 0, 175), (180, 60, 255))],
    "black": [((0, 0, 0), (180, 255, 55))],
    "purple": [((135, 50, 40), (169, 255, 255))],
}

FOUNDATIONPOSE_FILES = (
    "estimater.py",
    "learning/training/predict_pose_refine.py",
    "learning/training/predict_score.py",
)


@dataclass
class Detection:
    """One source-part estimate in the world frame."""

    part_type: str
    color: str
    score: float
    position: np.ndarray
    yaw_rad: float
    box_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    backend: str

    def as_dict(self, index: int) -> dict:
        return {
            "instance_key": f"vision-{index}",
            "type": self.part_type,
            "color": self.color,
            "score": round(float(self.score), 6),
            "position": [round(float(value), 7) for value in self.position],
            "yaw_rad": round(float(self.yaw_rad), 7),
            "box_xyxy": list(self.box_xyxy),
            "backend": self.backend,
        }


def _camera_point_to_world(
    point: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Convert ROS optical coordinates for the downward fixed camera to world."""
    if camera_rotation is not None:
        return camera_position + camera_rotation @ point
    return np.array(
        [
            camera_position[0] + point[0],
            camera_position[1] - point[1],
            camera_position[2] - point[2],
        ],
        dtype=float,
    )


def _masked_points(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    values = depth[rows, columns]
    valid = np.isfinite(values) & (values > 0.05) & (values < 3.0)
    rows = rows[valid]
    columns = columns[valid]
    values = values[valid]
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (columns.astype(np.float32) - cx) * values / fx
    y = (rows.astype(np.float32) - cy) * values / fy
    return np.column_stack((x, y, values)).astype(np.float32)


def _mask_color(rgb: np.ndarray, mask: np.ndarray) -> str:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    best_name = "unknown"
    best_count = 0
    for name, ranges in COLOR_RANGES.items():
        selected = np.zeros(mask.shape, dtype=np.uint8)
        for lower, upper in ranges:
            selected |= cv2.inRange(
                hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8)
            )
        count = int(np.count_nonzero((selected > 0) & mask))
        if count > best_count:
            best_count = count
            best_name = name
    return best_name


def _mask_type(mask: np.ndarray) -> tuple[str, float]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "brick_2x2", 1.0
    rectangle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    width, height = rectangle[1]
    ratio = max(width, height) / max(1.0, min(width, height))
    return ("brick_4x2" if ratio >= 1.45 else "brick_2x2"), float(ratio)


def _mask_yaw(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    corners = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    return float(math.atan2(-float(edge[1]), float(edge[0])))


def _box_for_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return (
        int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1
    )


class GeometryBackend:
    """Fast color and depth baseline for simulator integration tests."""

    name = "rgbd_geometry"

    def __init__(self, minimum_area_px: int = 18):
        self.minimum_area_px = int(minimum_area_px)

    def infer(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_position: np.ndarray,
        requested_parts: Iterable[dict],
        camera_rotation: np.ndarray | None = None,
    ) -> list[Detection]:
        requested = {
            (str(part.get("color", "unknown")), str(part.get("type", "brick_2x2")))
            for part in requested_parts
        }
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        detections: list[Detection] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for color, ranges in COLOR_RANGES.items():
            if requested and not any(item[0] == color for item in requested):
                continue
            color_mask = np.zeros(depth.shape, dtype=np.uint8)
            for lower, upper in ranges:
                color_mask |= cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                )
            color_mask[~np.isfinite(depth)] = 0
            if camera_rotation is None:
                # In the overhead view, depth separates a white part from the
                # white tabletop even when their RGB masks are connected.
                surface_z = camera_position[2] - depth
                color_mask[surface_z < 0.052] = 0
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if cv2.contourArea(contour) < self.minimum_area_px:
                    continue
                mask = np.zeros(depth.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
                part_type, ratio = _mask_type(mask > 0)
                if (requested and camera_rotation is None and
                        (color, part_type) not in requested):
                    continue
                points = _masked_points(mask > 0, depth, intrinsics)
                if points.shape[0] < self.minimum_area_px:
                    continue
                # The visible depth is the upper surface.  Convert it to the
                # body-centre convention expected by the executor.
                camera_point = np.median(points, axis=0)
                position = _camera_point_to_world(
                    camera_point, camera_position, camera_rotation
                )
                position[2] -= PART_HEIGHT_M[part_type] / 2.0
                if camera_rotation is not None:
                    world_points = (
                        points @ camera_rotation.T + camera_position
                    )
                    lower = np.percentile(world_points[:, :2], 2.0, axis=0)
                    upper = np.percentile(world_points[:, :2], 98.0, axis=0)
                    position[:2] = (lower + upper) / 2.0
                detections.append(
                    Detection(
                        part_type=part_type,
                        color=color,
                        score=min(0.99, 0.80 + 0.05 * ratio),
                        position=position,
                        yaw_rad=_mask_yaw(mask > 0),
                        box_xyxy=_box_for_mask(mask > 0),
                        mask=mask > 0,
                        backend=self.name,
                    )
                )
        return detections


class FoundationPoseBackend:
    """GroundingDINO, SAM and FoundationPose production backend."""

    name = "groundingdino_sam_foundationpose"

    def __init__(
        self,
        foundationpose_root: str,
        detector_model: str,
        sam_model: str,
        device: str,
        detection_threshold: float,
        register_iterations: int,
    ):
        if not foundationpose_root:
            raise RuntimeError(
                "foundationpose_root is empty; install FoundationPose separately and set "
                "FOUNDATIONPOSE_DIR or the ROS parameter"
            )
        root = Path(foundationpose_root).expanduser().resolve()
        if not (root / "estimater.py").exists():
            raise RuntimeError(f"FoundationPose estimater.py was not found in {root}")
        sys.path.insert(0, str(root))
        self.torch = importlib.import_module("torch")
        if not device.startswith("cuda"):
            raise RuntimeError("FoundationPose requires a CUDA device")
        if not self.torch.cuda.is_available():
            raise RuntimeError(
                "FoundationPose requires CUDA, but torch.cuda.is_available() is false"
            )
        transformers = importlib.import_module("transformers")
        self.device = device
        pipeline_device = 0 if device.startswith("cuda") else -1
        self.detector = transformers.pipeline(
            task="zero-shot-object-detection",
            model=detector_model,
            device=pipeline_device,
        )
        self.sam_processor = transformers.AutoProcessor.from_pretrained(sam_model)
        self.sam_model = transformers.AutoModelForMaskGeneration.from_pretrained(sam_model)
        self.sam_model = self.sam_model.to(device).eval()
        estimater = importlib.import_module("estimater")
        self.FoundationPose = estimater.FoundationPose
        self.scorer = estimater.ScorePredictor()
        self.refiner = estimater.PoseRefinePredictor()
        self.detection_threshold = float(detection_threshold)
        self.register_iterations = int(register_iterations)
        self.pose_estimators: dict[str, object] = {}

    @staticmethod
    def _mesh(part_type: str):
        import trimesh

        width = 0.064 if part_type == "brick_4x2" else 0.032
        depth = 0.032
        height = PART_HEIGHT_M[part_type]
        body = trimesh.creation.box(extents=(width, depth, height))
        studs = []
        columns = 4 if part_type == "brick_4x2" else 2
        for x_index in range(columns):
            for y_index in range(2):
                stud = trimesh.creation.cylinder(radius=0.0042, height=0.004, sections=24)
                stud.apply_translation(
                    (
                        (x_index - (columns - 1) / 2.0) * 0.016,
                        (y_index - 0.5) * 0.016,
                        height / 2.0 + 0.002,
                    )
                )
                studs.append(stud)
        return trimesh.util.concatenate([body, *studs])

    @staticmethod
    def _symmetry_transforms(part_type: str) -> np.ndarray:
        """Return the exact yaw symmetries of an untextured rectangular brick."""
        count = 4 if part_type == "brick_2x2" else 2
        transforms = []
        for index in range(count):
            angle = index * 2.0 * math.pi / count
            cosine, sine = math.cos(angle), math.sin(angle)
            transform = np.eye(4, dtype=np.float32)
            transform[:2, :2] = ((cosine, -sine), (sine, cosine))
            transforms.append(transform)
        return np.stack(transforms)

    def _estimator(self, part_type: str):
        if part_type in self.pose_estimators:
            return self.pose_estimators[part_type]
        mesh = self._mesh(part_type)
        # FoundationPose centres this mesh internally.  Passing vertices and
        # matching vertex normals follows its public API and avoids an empty
        # normal cloud in implementations that do not calculate normals.
        points = np.asarray(mesh.vertices, dtype=np.float32).copy()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
        estimator = self.FoundationPose(
            model_pts=points,
            model_normals=normals,
            symmetry_tfs=self._symmetry_transforms(part_type),
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
        )
        self.pose_estimators[part_type] = estimator
        return estimator

    def _sam_masks(self, rgb: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
        from PIL import Image

        image = Image.fromarray(rgb)
        inputs = self.sam_processor(images=image, input_boxes=[boxes], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]
        masks = masks.detach().cpu().numpy()
        return [(item[0] > 0.0) for item in masks]

    def infer(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_position: np.ndarray,
        requested_parts: Iterable[dict],
        camera_rotation: np.ndarray | None = None,
    ) -> list[Detection]:
        from PIL import Image

        parts = list(requested_parts)
        labels = sorted(
            {
                f"{part.get('color', '')} LEGO "
                f"{'four by two' if part.get('type') == 'brick_4x2' else 'two by two'} brick"
                for part in parts
            }
        )
        raw = self.detector(
            Image.fromarray(rgb),
            candidate_labels=labels + ["LEGO brick"],
            threshold=self.detection_threshold,
        )
        boxes = [
            [int(item["box"][key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            for item in raw
        ]
        if not boxes:
            return []
        masks = self._sam_masks(rgb, boxes)
        detections: list[Detection] = []
        for item, box, mask in zip(raw, boxes, masks):
            if int(np.count_nonzero(mask)) < 12:
                continue
            color = _mask_color(rgb, mask)
            part_type, _ = _mask_type(mask)
            if not any(
                str(part.get("color")) == color and str(part.get("type")) == part_type
                for part in parts
            ):
                continue
            estimator = self._estimator(part_type)
            transform = estimator.register(
                K=np.ascontiguousarray(intrinsics, dtype=np.float32),
                rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
                depth=np.ascontiguousarray(depth, dtype=np.float32),
                ob_mask=np.ascontiguousarray(mask, dtype=bool),
                iteration=self.register_iterations,
            )
            if transform is None:
                continue
            transform = np.asarray(transform, dtype=float)
            world_position = _camera_point_to_world(
                transform[:3, 3], camera_position, camera_rotation
            )
            # Convert the estimated camera-frame orientation into world axes.
            optical_to_world = (
                np.diag([1.0, -1.0, -1.0])
                if camera_rotation is None
                else camera_rotation
            )
            world_rotation = optical_to_world @ transform[:3, :3]
            yaw = math.atan2(world_rotation[1, 0], world_rotation[0, 0])
            detections.append(
                Detection(
                    part_type=part_type,
                    color=color,
                    score=float(item.get("score", 0.0)),
                    position=world_position,
                    yaw_rad=float(yaw),
                    box_xyxy=tuple(box),
                    mask=mask,
                    backend=self.name,
                )
            )
        return detections


def make_backend(name: str, **settings):
    if name == "geometry":
        return GeometryBackend(minimum_area_px=settings.get("minimum_area_px", 18))
    if name == "foundationpose":
        return FoundationPoseBackend(
            foundationpose_root=settings.get(
                "foundationpose_root", os.environ.get("FOUNDATIONPOSE_DIR", "")
            ),
            detector_model=settings.get(
                "detector_model", "IDEA-Research/grounding-dino-tiny"
            ),
            sam_model=settings.get("sam_model", "facebook/sam-vit-huge"),
            device=settings.get("device", "cuda"),
            detection_threshold=settings.get("detection_threshold", 0.20),
            register_iterations=settings.get("register_iterations", 5),
        )
    raise ValueError(f"unknown vision backend: {name}")


def foundationpose_preflight(foundationpose_root: str, device: str = "cuda") -> dict:
    """Check the local production-vision installation without loading model weights."""
    root = Path(foundationpose_root).expanduser().resolve() if foundationpose_root else None
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record(
        "foundationpose_root",
        root is not None and root.is_dir(),
        str(root) if root is not None else "FOUNDATIONPOSE_DIR is not set",
    )
    for relative_path in FOUNDATIONPOSE_FILES:
        path = root / relative_path if root is not None else None
        record(
            relative_path,
            path is not None and path.is_file(),
            str(path) if path is not None else "FoundationPose root is unavailable",
        )

    torch_spec = importlib.util.find_spec("torch")
    record("torch", torch_spec is not None, "installed" if torch_spec else "not installed")
    cuda_available = False
    cuda_detail = "PyTorch is unavailable"
    if torch_spec is not None:
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            cuda_detail = (
                f"torch={torch.__version__}, build_cuda={torch.version.cuda}, "
                f"available={cuda_available}"
            )
        except Exception as error:  # pragma: no cover - depends on local binary install
            cuda_detail = f"PyTorch import failed: {error}"
    record("cuda", device.startswith("cuda") and cuda_available, cuda_detail)

    for package in ("transformers", "PIL", "trimesh", "cv2"):
        found = importlib.util.find_spec(package) is not None
        record(package, found, "installed" if found else "not installed")

    foundationpose_imported = False
    foundationpose_detail = "FoundationPose root is unavailable"
    if root is not None and (root / "estimater.py").is_file():
        inserted = str(root) not in sys.path
        if inserted:
            sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("estimater")
            required = ("FoundationPose", "ScorePredictor", "PoseRefinePredictor")
            missing = [name for name in required if not hasattr(module, name)]
            foundationpose_imported = not missing
            foundationpose_detail = (
                "estimater import succeeded"
                if not missing
                else "missing exports: " + ", ".join(missing)
            )
        except Exception as error:  # pragma: no cover - external CUDA installation
            foundationpose_detail = f"estimater import failed: {error}"
        finally:
            if inserted:
                sys.path.remove(str(root))
    record("foundationpose_import", foundationpose_imported, foundationpose_detail)

    return {
        "ready": all(item["passed"] for item in checks),
        "device": device,
        "foundationpose_root": str(root) if root is not None else "",
        "checks": checks,
        "note": (
            "This preflight does not download model weights or run inference. "
            "A complete check still requires a live RGB-D episode."
        ),
    }


def main(argv=None) -> int:
    """Run a lightweight readiness check for the CUDA perception backend."""
    import argparse

    parser = argparse.ArgumentParser(description="Check LEGO vision dependencies")
    parser.add_argument(
        "--foundationpose-root",
        default=os.environ.get("FOUNDATIONPOSE_DIR", ""),
        help="path to an installed FoundationPose checkout",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    report = foundationpose_preflight(arguments.foundationpose_root, arguments.device)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
