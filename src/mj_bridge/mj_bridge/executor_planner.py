"""Dependency-aware assembly ordering for benchmark executor baselines."""

from __future__ import annotations

from collections import defaultdict

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


def _grasp_spin(block: dict, blocks: list[dict]) -> float:
    """Preserve the original planner's 90-degree-first accessibility policy."""
    x, y, z = (float(v) for v in block["target"]["position"])
    same_level = [
        other
        for other in blocks
        if other["id"] != block["id"]
        and abs(float(other["target"]["position"][2]) - z) < 0.01
    ]
    x_blocked = any(
        abs(float(other["target"]["position"][1]) - y) < 0.005
        and abs(float(other["target"]["position"][0]) - x) < 0.045
        for other in same_level
    )
    if not x_blocked:
        return 90.0
    y_blocked = any(
        abs(float(other["target"]["position"][0]) - x) < 0.005
        and abs(float(other["target"]["position"][1]) - y) < 0.045
        for other in same_level
    )
    return 0.0 if not y_blocked else 90.0


def plan_assembly(product: dict, registry: dict | None = None) -> dict:
    """Create a stable bottom-up plan while retaining explicit product IDs."""
    registry = registry or load_registry()
    validate_product(product, registry)
    blocks = product["blocks"]
    by_id = {block["id"]: block for block in blocks}
    dependencies = _support_graph(product, registry)
    remaining = set(by_id)
    completed: set[str] = set()
    ordered: list[dict] = []

    while remaining:
        ready = [block_id for block_id in remaining if dependencies[block_id] <= completed]
        if not ready:
            raise ValueError("product contains an unresolved or cyclic support relationship")
        ready.sort(key=lambda block_id: (
            float(by_id[block_id]["target"]["position"][2]),
            blocks.index(by_id[block_id]),
            block_id,
        ))
        block_id = ready[0]
        block = by_id[block_id]
        ordered.append({
            "id": block_id,
            "type": block["type"],
            "color": block.get("color", "gray"),
            "grasp_spin_deg": _grasp_spin(block, blocks),
            "depends_on": sorted(dependencies[block_id]),
        })
        remaining.remove(block_id)
        completed.add(block_id)

    return {
        "schema_version": 1,
        "product": product["product"],
        "steps": ordered,
    }
