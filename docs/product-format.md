# Product format

The canonical product format is YAML and uses `schema_version: 1`. The machine-readable schema is stored in `src/mj_bridge/mj_bridge/product_schema.json`.

```yaml
schema_version: 1
product:
  name: bridge
blocks:
  - id: left_support
    type: brick_2x2
    color: blue
    target: {position: [-0.032, 0.0, 0.0], yaw_deg: 0}
  - id: right_support
    type: brick_2x2
    color: blue
    target: {position: [0.032, 0.0, 0.0], yaw_deg: 0}
  - id: top_beam
    type: brick_4x2
    color: white
    target: {position: [0.0, 0.0, 0.0192], yaw_deg: 0}
```

## Fields

- `product.name` identifies the product and is included in episode IDs.
- `blocks[].id` is the stable identity used for spawning and scoring.
- `blocks[].type` must exist in `part_registry.yaml`.
- `blocks[].color` selects a named display color; unknown values use gray.
- `target.position` is `[x, y, z]` in metres relative to the assembly origin.
- `target.yaw_deg` is rotation about world Z in degrees.

The target position refers to the lowest placement level before the simulator adds the assembly plate and part-center offsets. Higher layers are expressed as positive Z offsets.

## Legacy conversion

The CLI accepts the original `blocks + pos + rotation` form and planner exports using `tasks + place`. Legacy files may mix degrees and fractional-radian yaw values. Duplicate legacy names receive deterministic unique IDs.

```bash
lego-bench convert old_product.yaml product_v1.yaml
lego-bench validate product_v1.yaml
```

The compatibility layer is intended for migration. New datasets should use schema version 1.

## Adding a part type

A new part requires a unique registry key, physical dimensions, stud layout, mass, yaw symmetry, and a matching primitive or licensed geometry implementation in the scene builder. Add tests for placement bounds, deterministic generation, scene loading, and symmetry-aware scoring.
