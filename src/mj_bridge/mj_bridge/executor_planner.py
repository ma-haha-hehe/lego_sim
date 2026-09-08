"""Dependency-aware assembly ordering for benchmark executor baselines."""

from __future__ import annotations

from collections import defaultdict
from math import cos, radians, sin

from .benchmark_core import load_registry, validate_product


def _footprint(block: dict, registry: dict) -> tuple[float, float]:
    size = registry[block["type"]]["size_m"]
    yaw = int(round(float(block["target"].get("yaw_deg", 0.0)))) % 180
    return (float(size[1]), float(size[0])) if yaw == 90 else (float(size[0]), float(size[1]))


def _overlap_xy(lower: dict, upper: dict, registry: dict, tolerance: float = 0.002) -> bool:
    lx, ly, _ = lower["target"]["position"]
    ux, uy, _ = upper["target"]["position"]
    lw, ld = _footprint(lower, registry)
    uw, ud = _footprint(upper, registry)
    return (
        abs(float(lx) - float(ux)) < (lw + uw) / 2.0 - tolerance
        and abs(float(ly) - float(uy)) < (ld + ud) / 2.0 - tolerance
    )


def _support_graph(product: dict, registry: dict) -> dict[str, set[str]]:
    """Return direct lower-block dependencies for every product block."""
    dependencies: dict[str, set[str]] = defaultdict(set)
    blocks = product["blocks"]
    for upper in blocks:
        upper_z = float(upper["target"]["position"][2])
        candidates = [
            lower
            for lower in blocks
            if float(lower["target"]["position"][2]) < upper_z
            and _overlap_xy(lower, upper, registry)
        ]
        if not candidates:
            continue
        closest_z = max(float(block["target"]["position"][2]) for block in candidates)
        for lower in candidates:
            if abs(float(lower["target"]["position"][2]) - closest_z) <= 1e-6:
                dependencies[upper["id"]].add(lower["id"])
    return dependencies


def _local_offset(block: dict, other: dict) -> tuple[float, float]:
    """Return another block's offset in the candidate block's XY frame."""
    x, y, _ = (float(value) for value in block["target"]["position"])
    other_x, other_y, _ = (
        float(value) for value in other["target"]["position"]
    )
    yaw = radians(float(block["target"].get("yaw_deg", 0.0)))
    dx = other_x - x
    dy = other_y - y
    return cos(yaw) * dx + sin(yaw) * dy, -sin(yaw) * dx + cos(yaw) * dy


def _grasp_accessible(block: dict, scene_blocks: list[dict], spin_deg: float) -> bool:
    """Check the selected finger corridor in the block's local frame."""
    z = float(block["target"]["position"][2])
    for other in scene_blocks:
        if other["id"] == block["id"]:
            continue
        if abs(float(other["target"]["position"][2]) - z) >= 0.01:
            continue
        local_x, local_y = _local_offset(block, other)
        if spin_deg == 90.0:
            blocked = abs(local_y) < 0.005 and abs(local_x) < 0.045
        else:
            blocked = abs(local_x) < 0.005 and abs(local_y) < 0.045
        if blocked:
            return False
    return True


def _grasp_spin(block: dict, scene_blocks: list[dict]) -> float:
    """Choose 90 then 0 degrees relative to the block's own frame."""
    if _grasp_accessible(block, scene_blocks, 90.0):
        return 90.0
    if _grasp_accessible(block, scene_blocks, 0.0):
        return 0.0
    return 90.0


def plan_assembly(product: dict, registry: dict | None = None) -> dict:
    """Plan reverse disassembly, then invert it into a stable assembly order."""
    registry = registry or load_registry()
    validate_product(product, registry)
    blocks = product["blocks"]
    by_id = {block["id"]: block for block in blocks}
    dependencies = _support_graph(product, registry)
    input_index = {block["id"]: index for index, block in enumerate(blocks)}
    remaining = set(by_id)
    disassembly: list[tuple[str, float]] = []

    while remaining:
        removable = [
            block_id
            for block_id in remaining
            if not any(
                block_id in dependencies[other_id]
                for other_id in remaining
                if other_id != block_id
            )
        ]
        if not removable:
            raise ValueError("product contains an unresolved or cyclic support relationship")
        removable.sort(key=lambda block_id: (
            -float(by_id[block_id]["target"]["position"][2]),
            input_index[block_id],
            block_id,
        ))
        scene_blocks = [by_id[block_id] for block_id in remaining]
        selected_id = removable[0]
        selected_spin = 90.0
        for block_id in removable:
            block = by_id[block_id]
            spin = _grasp_spin(block, scene_blocks)
            if _grasp_accessible(block, scene_blocks, spin):
                selected_id = block_id
                selected_spin = spin
                break
        disassembly.append((selected_id, selected_spin))
        remaining.remove(selected_id)

    ordered: list[dict] = []
    for block_id, grasp_spin in reversed(disassembly):
        block = by_id[block_id]
        ordered.append({
            "id": block_id,
            "type": block["type"],
            "color": block.get("color", "gray"),
            "grasp_spin_deg": grasp_spin,
            "depends_on": sorted(dependencies[block_id]),
        })

    return {
        "schema_version": 1,
        "product": product["product"],
        "planning_strategy": "reverse_disassembly_90_first",
        "grasp_angle_frame": "part_local",
        "steps": ordered,
    }
