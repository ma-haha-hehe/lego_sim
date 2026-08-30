# scene_builder.py
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import yaml
import numpy as np

try:
    from .benchmark_core import load_yaml as load_benchmark_yaml
except ImportError:  # Allow `python scene_builder.py` during local development.
    from benchmark_core import load_yaml as load_benchmark_yaml


# ============================================================

# ============================================================

BASE_DIR = os.path.dirname(__file__)
PART_REGISTRY = load_benchmark_yaml(os.path.join(BASE_DIR, "part_registry.yaml"))["parts"]

INITIAL_YAML = os.path.join(BASE_DIR, "initial.yaml")
SCENE_TEMPLATE_XML = os.path.join(BASE_DIR, "scene_template.xml")
SCENE_OUTPUT_XML = os.path.join(BASE_DIR, "scene.xml")


# ============================================================

# ============================================================


# <body name="table" pos="0.40 0 0.02">
#   <geom type="box" size="0.30 0.50 0.02"/>
# </body>
#

# table center z 0.02 + table half height 0.02 = 0.04
TABLE_TOP_Z = 0.04


BRICK_BODY_HALF_HEIGHT = 0.0095


STUD_HALF_HEIGHT = 0.0020



# 0.04 + 0.0095 = 0.0495
BRICK_CENTER_Z = TABLE_TOP_Z + BRICK_BODY_HALF_HEIGHT


DESIRED_PARTS_CENTER = np.array([0.52, -0.12, BRICK_CENTER_Z], dtype=float)


EXTRA_OFFSET = np.array([0.00, 0.00, 0.00], dtype=float)


# ============================================================

# ============================================================


#


#


COLLISION_Z_OFFSET = -0.009


# ============================================================

# ============================================================


DENSITY = 150





# Sliding, torsional and rolling friction for the ABS-like part surfaces.
# The values are paired with the rubber fingertip pads in panda.xml and keep a
# physically grasped part from creeping during fast transfer motions.
BRICK_FRICTION = "7.0 0.4 0.04"
SUPPORT_FRICTION = "5.0 0.2 0.02"




# A two-timestep contact time constant and near-rigid impedance keep lightweight
# parts on the support surface when a position-controlled arm presses on them.
# Softer values allow visibly large penetration before producing enough force.
SOLREF = "0.002 1"
SOLIMP = "0.995 0.9999 0.0001"



VISUAL_CONTYPE = "0"
VISUAL_CONAFFINITY = "0"


# ============================================================

# ============================================================


STUD_PITCH = 0.016

# ============================================================

# ============================================================

ADD_ASSEMBLY_BASE_PLATE = True

ASSEMBLY_BASE_NAME = "assembly_base_plate"


# ASSEMBLY_ORIGIN_X = 0.25
# ASSEMBLY_ORIGIN_Y = 0.0
ASSEMBLY_BASE_CENTER_X = 0.35
ASSEMBLY_BASE_CENTER_Y = 0.35

BASE_PLATE_STUDS_X = 12
BASE_PLATE_STUDS_Y = 12

# 12 * 0.016 = 0.192 m
BASE_PLATE_HALF_X = BASE_PLATE_STUDS_X * STUD_PITCH / 2.0
BASE_PLATE_HALF_Y = BASE_PLATE_STUDS_Y * STUD_PITCH / 2.0


BASE_PLATE_THICKNESS = 0.006
BASE_PLATE_HALF_HEIGHT = BASE_PLATE_THICKNESS / 2.0


BASE_PLATE_CENTER_Z = TABLE_TOP_Z + BASE_PLATE_HALF_HEIGHT


BASE_STUD_RADIUS = 0.0042
BASE_STUD_HALF_HEIGHT = 0.0020




BASE_STUD_CENTER_Z_LOCAL = BASE_PLATE_HALF_HEIGHT + BASE_STUD_HALF_HEIGHT



STUD_RADIUS = 0.0042





BRICK_SPECS = {
    "brick_2x2": {
        "mesh": "lego_2x2",
        "studs_x": 2,
        "studs_y": 2,
        "body_half_size": np.array([0.016, 0.016, BRICK_BODY_HALF_HEIGHT], dtype=float),
        "mass_kg": float(PART_REGISTRY["brick_2x2"]["mass_kg"]),
    },
    "brick_4x2": {
        "mesh": "lego_2x4",
        "studs_x": 4,
        "studs_y": 2,
        "body_half_size": np.array([0.032, 0.016, BRICK_BODY_HALF_HEIGHT], dtype=float),
        "mass_kg": float(PART_REGISTRY["brick_4x2"]["mass_kg"]),
    },
}


# ============================================================

# ============================================================

COLOR_MAP = {
    "red": "1 0 0 1",
    "green": "0 1 0 1",
    "yellow": "1 1 0 1",
    "blue": "0 0 1 1",
    "gray": "0.5 0.5 0.5 1",
    "grey": "0.5 0.5 0.5 1",
    "white": "1 1 1 1",
    "black": "0.1 0.1 0.1 1",
    "orange": "1.0 0.45 0.0 1",
    "purple": "0.55 0.2 0.75 1",
}


# ============================================================

# ============================================================

def prettify_xml(elem: ET.Element) -> str:
    """Return a consistently formatted XML document."""
    rough = ET.tostring(elem, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def load_yaml(path: str) -> dict:
    """Load a YAML mapping from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_xml(path: str) -> ET.ElementTree:
    """Parse an XML document from disk."""
    return ET.parse(path)


def get_worldbody(root: ET.Element) -> ET.Element:
    """Return the scene worldbody element."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("scene_template.xml does not contain a worldbody element")
    return worldbody


def find_and_remove_old_dynamic_bricks(worldbody: ET.Element) -> None:
    """Remove dynamic parts and the assembly plate left by an earlier build."""
    to_remove = []

    for child in list(worldbody):
        if child.tag != "body":
            continue

        name = child.get("name", "")

        if "brick" in name or name.startswith("lego_figure"):
            to_remove.append(child)

        if name == ASSEMBLY_BASE_NAME:
            to_remove.append(child)

    for child in to_remove:
        worldbody.remove(child)

def quat_from_yaw(yaw_rad: float) -> str:
    """Return a MuJoCo wxyz quaternion for a rotation about the Z axis."""
    w = math.cos(yaw_rad / 2.0)
    z = math.sin(yaw_rad / 2.0)
    return f"{w:.10f} 0 0 {z:.10f}"


def get_rgba(color_name: str) -> str:
    """Return an RGBA value for a named color."""
    return COLOR_MAP.get(color_name.lower(), "0.7 0.7 0.7 1")


def local_to_world(block_pos, yaml_parts_center) -> np.ndarray:
    """Map a legacy part position into the MuJoCo world frame."""
    local = np.array(block_pos, dtype=float)

    world = local - yaml_parts_center + DESIRED_PARTS_CENTER + EXTRA_OFFSET

    return world


# ============================================================

# ============================================================

def set_common_collision_params(geom: ET.Element, density: float | None = None) -> None:
    """Apply shared contact parameters to a collision geometry."""
    geom.set("rgba", "0 0 0 0")
    geom.set("friction", BRICK_FRICTION)
    geom.set("solref", SOLREF)
    geom.set("solimp", SOLIMP)

    if density is not None:
        geom.set("density", str(density))


def add_duplo_collision_geoms(body: ET.Element, brick_type: str) -> None:
    """Add a hollow brick shell and cylindrical studs."""

    if brick_type not in BRICK_SPECS:
        raise ValueError(f"Unsupported brick type: {brick_type}")

    spec = BRICK_SPECS[brick_type]

    body_half = spec["body_half_size"]
    studs_x = spec["studs_x"]
    studs_y = spec["studs_y"]

    # Use the measured part mass from the public registry instead of deriving
    # mass from the deliberately simplified collision primitives.  A box
    # approximation is sufficient for the principal inertia of these small,
    # nearly symmetric parts and keeps the centre of mass aligned with the
    # collision body.
    mass = float(spec["mass_kg"])
    full_size = 2.0 * body_half
    inertia = mass / 12.0 * np.array([
        full_size[1] ** 2 + full_size[2] ** 2,
        full_size[0] ** 2 + full_size[2] ** 2,
        full_size[0] ** 2 + full_size[1] ** 2,
    ])
    ET.SubElement(body, "inertial", {
        "pos": f"0 0 {COLLISION_Z_OFFSET:.4f}",
        "mass": f"{mass:.6f}",
        "diaginertia": " ".join(f"{value:.10g}" for value in inertia),
    })


    # A solid box makes every upper brick rest on top of the studs and adds
    # roughly 4 mm to each layer.  Four perimeter walls and a top plate leave
    # the underside open, so studs can enter the cavity as they do on a real
    # brick.  The explicit inertial above owns the mass calculation.
    wall_thickness = 0.0025
    top_thickness = 0.0025

    def add_shell_box(size, pos):
        geom = ET.SubElement(body, "geom", {"type": "box"})
        geom.set("size", " ".join(f"{value:.4f}" for value in size))
        geom.set("pos", " ".join(f"{value:.4f}" for value in pos))
        set_common_collision_params(geom)

    add_shell_box(
        [wall_thickness / 2.0, body_half[1], body_half[2]],
        [body_half[0] - wall_thickness / 2.0, 0.0, COLLISION_Z_OFFSET],
    )
    add_shell_box(
        [wall_thickness / 2.0, body_half[1], body_half[2]],
        [-body_half[0] + wall_thickness / 2.0, 0.0, COLLISION_Z_OFFSET],
    )
    add_shell_box(
        [body_half[0] - wall_thickness, wall_thickness / 2.0, body_half[2]],
        [0.0, body_half[1] - wall_thickness / 2.0, COLLISION_Z_OFFSET],
    )
    add_shell_box(
        [body_half[0] - wall_thickness, wall_thickness / 2.0, body_half[2]],
        [0.0, -body_half[1] + wall_thickness / 2.0, COLLISION_Z_OFFSET],
    )
    add_shell_box(
        [body_half[0] - wall_thickness, body_half[1] - wall_thickness,
         top_thickness / 2.0],
        [0.0, 0.0,
         COLLISION_Z_OFFSET + body_half[2] - top_thickness / 2.0],
    )


    start_x = -(studs_x - 1) * STUD_PITCH / 2.0
    start_y = -(studs_y - 1) * STUD_PITCH / 2.0

    stud_center_z = BRICK_BODY_HALF_HEIGHT + STUD_HALF_HEIGHT

    for ix in range(studs_x):
        for iy in range(studs_y):
            x = start_x + ix * STUD_PITCH
            y = start_y + iy * STUD_PITCH

            geom_stud = ET.SubElement(body, "geom")
            geom_stud.set("type", "cylinder")

            # cylinder size: radius half-height
            geom_stud.set(
                "size",
                f"{STUD_RADIUS:.4f} {STUD_HALF_HEIGHT:.4f}",
            )


            geom_stud.set(
                "pos",
                f"{x:.4f} {y:.4f} {stud_center_z + COLLISION_Z_OFFSET:.4f}",
            )


            set_common_collision_params(geom_stud, density=30)


def create_assembly_base_plate() -> ET.Element:
    """Create the fixed 12 by 12 stud assembly plate."""

    body = ET.Element("body")
    body.set("name", ASSEMBLY_BASE_NAME)
    body.set(
        "pos",
        f"{ASSEMBLY_BASE_CENTER_X:.4f} {ASSEMBLY_BASE_CENTER_Y:.4f} {BASE_PLATE_CENTER_Z:.4f}",
    )


    geom_base = ET.SubElement(body, "geom")
    geom_base.set("name", "assembly_base_plate_body")
    geom_base.set("type", "box")
    geom_base.set(
        "size",
        f"{BASE_PLATE_HALF_X:.4f} {BASE_PLATE_HALF_Y:.4f} {BASE_PLATE_HALF_HEIGHT:.4f}",
    )
    geom_base.set("rgba", "0.12 0.12 0.12 1")
    geom_base.set("friction", SUPPORT_FRICTION)
    geom_base.set("solref", SOLREF)
    geom_base.set("solimp", SOLIMP)

    # ---------- 12x12 studs ----------
    start_x = -(BASE_PLATE_STUDS_X - 1) * STUD_PITCH / 2.0
    start_y = -(BASE_PLATE_STUDS_Y - 1) * STUD_PITCH / 2.0

    for ix in range(BASE_PLATE_STUDS_X):
        for iy in range(BASE_PLATE_STUDS_Y):
            x = start_x + ix * STUD_PITCH
            y = start_y + iy * STUD_PITCH

            geom_stud = ET.SubElement(body, "geom")
            geom_stud.set("name", f"assembly_base_stud_{ix}_{iy}")
            geom_stud.set("type", "cylinder")
            geom_stud.set(
                "size",
                f"{BASE_STUD_RADIUS:.4f} {BASE_STUD_HALF_HEIGHT:.4f}",
            )
            geom_stud.set(
                "pos",
                f"{x:.4f} {y:.4f} {BASE_STUD_CENTER_Z_LOCAL:.4f}",
            )
            geom_stud.set("rgba", "0.12 0.12 0.12 1")
            geom_stud.set("friction", SUPPORT_FRICTION)
            geom_stud.set("solref", SOLREF)
            geom_stud.set("solimp", SOLIMP)

    return body

# ============================================================

# ============================================================

def create_brick_body(block: dict, yaml_parts_center: np.ndarray) -> ET.Element:
    """Create a movable legacy brick with mesh visuals and primitive collision."""

    name = block["name"]
    brick_type = block["type"]
    color = block.get("color", "gray")
    pos = block["pos"]
    rotation = block.get("rotation", [0.0, 0.0, 0.0])

    if brick_type not in BRICK_SPECS:
        raise ValueError(f"Unsupported brick type: {brick_type}")

    spec = BRICK_SPECS[brick_type]
    mesh_name = spec["mesh"]
    rgba = get_rgba(color)

    yaw = float(rotation[2])
    world_pos = local_to_world(pos, yaml_parts_center)
    quat = quat_from_yaw(yaw)

    body = ET.Element("body")
    body.set("name", name)
    body.set(
        "pos",
        f"{world_pos[0]:.4f} {world_pos[1]:.4f} {world_pos[2]:.4f}",
    )
    body.set("quat", quat)


    ET.SubElement(body, "freejoint")

    # ---------- visual mesh ----------
    geom_visual = ET.SubElement(body, "geom")
    geom_visual.set("type", "mesh")
    geom_visual.set("mesh", mesh_name)
    geom_visual.set("rgba", rgba)


    geom_visual.set("contype", VISUAL_CONTYPE)
    geom_visual.set("conaffinity", VISUAL_CONAFFINITY)

    # ---------- collision ----------
    add_duplo_collision_geoms(body, brick_type)

    return body


def create_episode_brick_body(block: dict) -> ET.Element:
    """Create one loose source part from a benchmark episode manifest."""
    brick_type = block["type"]
    if brick_type not in BRICK_SPECS:
        raise ValueError(f"Unsupported brick type: {brick_type}")
    body = ET.Element("body", {
        "name": block["body_name"],
        "quat": quat_from_yaw(float(block.get("yaw_rad", 0.0))),
    })
    ET.SubElement(body, "freejoint")
    add_duplo_collision_geoms(body, brick_type)
    # Public episodes render the same self-authored primitives used for collision.
    # This keeps release artifacts independent from unverified LEGO mesh files.
    rgba = get_rgba(block.get("color", "gray"))
    for geom in body.findall("geom"):
        geom.set("rgba", rgba)
    p = block["position"]
    body.set("pos", f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}")
    return body


def add_figure_geom(body, *, name, geom_type, size, pos=None, rgba="0.8 0.8 0.8 1", **attrs):
    """Add a primitive geometry that participates in rendering and collision."""
    geom = ET.SubElement(body, "geom")
    geom.set("name", name)
    geom.set("type", geom_type)
    geom.set("size", size)
    if pos is not None:
        geom.set("pos", pos)
    geom.set("rgba", rgba)
    geom.set("friction", "1.2 0.08 0.005")
    geom.set("solref", "0.006 1")
    geom.set("solimp", "0.92 0.98 0.002")
    geom.set("density", "650")
    for key, value in attrs.items():
        geom.set(key, str(value))
    return geom


def create_lego_figure_body(spec: dict) -> ET.Element:
    """Create the legacy single-body minifigure benchmark target."""
    name = spec.get("name", "lego_figure_1")
    x, y, yaw = [float(v) for v in spec.get("pose", [0.48, -0.12, 0.0])]

    body = ET.Element("body")
    body.set("name", name)
    body.set("pos", f"{x:.5f} {y:.5f} {TABLE_TOP_Z:.5f}")
    body.set("quat", quat_from_yaw(yaw))
    ET.SubElement(body, "freejoint")


    add_figure_geom(body, name=f"{name}_left_leg", geom_type="box",
                    size="0.0038 0.0040 0.0060", pos="-0.0043 0 0.0060",
                    rgba="0.12 0.22 0.75 1")
    add_figure_geom(body, name=f"{name}_right_leg", geom_type="box",
                    size="0.0038 0.0040 0.0060", pos="0.0043 0 0.0060",
                    rgba="0.12 0.22 0.75 1")
    add_figure_geom(body, name=f"{name}_hip", geom_type="box",
                    size="0.0082 0.0042 0.0022", pos="0 0 0.0138",
                    rgba="0.12 0.22 0.75 1")
    add_figure_geom(body, name=f"{name}_torso", geom_type="box",
                    size="0.0090 0.0045 0.0070", pos="0 0 0.0230",
                    rgba="0.82 0.10 0.08 1")
    add_figure_geom(body, name=f"{name}_left_arm", geom_type="capsule",
                    size="0.0022", rgba="0.82 0.10 0.08 1",
                    fromto="-0.0105 0 0.0280 -0.0125 0 0.0170")
    add_figure_geom(body, name=f"{name}_right_arm", geom_type="capsule",
                    size="0.0022", rgba="0.82 0.10 0.08 1",
                    fromto="0.0105 0 0.0280 0.0125 0 0.0170")
    add_figure_geom(body, name=f"{name}_head", geom_type="cylinder",
                    size="0.0052 0.0052", pos="0 0 0.0352",
                    rgba="1.0 0.72 0.18 1")
    add_figure_geom(body, name=f"{name}_head_stud", geom_type="cylinder",
                    size="0.0030 0.0012", pos="0 0 0.0416",
                    rgba="1.0 0.72 0.18 1")
    return body


# ============================================================

# ============================================================

def build(
    initial_yaml: str = INITIAL_YAML,
    template_xml: str = SCENE_TEMPLATE_XML,
    output_xml: str = SCENE_OUTPUT_XML,
    episode_manifest: str | None = None,
) -> str:
    """Build a standalone MuJoCo scene from a template and episode manifest."""

    if not os.path.exists(initial_yaml):
        raise FileNotFoundError(f"Initial scene configuration not found: {initial_yaml}")

    if not os.path.exists(template_xml):
        raise FileNotFoundError(f"Scene template not found: {template_xml}")

    data = load_yaml(initial_yaml)

    tree = parse_xml(template_xml)
    root = tree.getroot()
    worldbody = get_worldbody(root)


    find_and_remove_old_dynamic_bricks(worldbody)

    if ADD_ASSEMBLY_BASE_PLATE:
        base_plate = create_assembly_base_plate()
        worldbody.append(base_plate)


    parts_plate = data.get("parts_plate", {})
    yaml_parts_center = np.array(
        parts_plate.get("center", [0.0, 0.0, 0.0]),
        dtype=float,
    )

    if episode_manifest:
        episode = load_benchmark_yaml(episode_manifest)
        asset = root.find("asset")
        if asset is not None:
            for mesh in list(asset.findall("mesh")):
                if mesh.get("name", "").startswith("lego_"):
                    asset.remove(mesh)
        blocks = episode.get("spawned_blocks", [])
        figures = []
        for block in blocks:
            worldbody.append(create_episode_brick_body(block))
    else:
        blocks = data.get("blocks", [])
        for block in blocks:
            worldbody.append(create_brick_body(block, yaml_parts_center))
        figures = data.get("figures", [])
        for figure in figures:
            worldbody.append(create_lego_figure_body(figure))

    xml_text = prettify_xml(root)

    with open(output_xml, "w", encoding="utf-8") as f:
        f.write(xml_text)

    print("scene generated")
    print(f"template : {template_xml}")
    print(f"initial  : {initial_yaml}")
    print(f"output   : {output_xml}")
    print(f"blocks   : {len(blocks)}")
    print(f"figures  : {len(figures)}")
    print(f"parts center mapped to world: {DESIRED_PARTS_CENTER.tolist()}")
    print(f"collision z offset: {COLLISION_Z_OFFSET}")
    print(f"brick body half height: {BRICK_BODY_HALF_HEIGHT}")
    print(f"stud radius: {STUD_RADIUS}")
    print(f"brick friction: {BRICK_FRICTION}")
    print(f"support friction: {SUPPORT_FRICTION}")
    print(f"solref: {SOLREF}")
    print(f"solimp: {SOLIMP}")

    return output_xml


if __name__ == "__main__":
    build()
