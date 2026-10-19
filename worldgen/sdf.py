# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDF 1.9 writer with a design-ledger header (T0 §6).

The world header (physics, system plugins, scene and light) is copied verbatim from the benchmark's Corridor
world, so new worlds load like the current ones. The copied lines are kept in `corridor_header.sdf` so the
collection builds without the benchmark repository.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from . import WORLDGEN_VERSION
from .world import MATERIALS, Kind, Primitive, World, sorted_primitives

HEADER_FILE = Path(__file__).with_name("corridor_header.sdf")
# Provenance of HEADER_FILE: the benchmark repository's Corridor world and that file's SHA-256.
HEADER_SOURCE = "simulation/worlds/corridor/corridor.world"
HEADER_SOURCE_SHA256 = "2b52a0ad88202c2f2b72f39a9101340ae9b9443c4245356c7a2c6ea11a5c07de"
FLOOR_MARGIN_M = 2.0
FLOOR_THICKNESS_M = 0.05


def m(value: float) -> str:
    """Metres with four fixed decimals (negative zero printed as zero)."""
    text = f"{value:.4f}"
    return "0.0000" if text == "-0.0000" else text


def rad(value: float) -> str:
    text = f"{value:.6f}"
    return "0.000000" if text == "-0.000000" else text


def header_block() -> tuple[str, str]:
    """The Corridor world's lines from `<gravity>` to the end of its first `<light>`, and the source file's
    SHA-256."""
    return HEADER_FILE.read_text(encoding="utf-8").removesuffix("\n"), HEADER_SOURCE_SHA256


def split_after_plugins(header: str) -> tuple[str, str]:
    """The header up to and including the end of its last `<plugin>` element, and the lines after it (the
    newline between the two parts belongs to neither). E worlds insert the E0 system plugin between them."""
    lines = header.split("\n")
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("<plugin")]
    if not starts:
        raise ValueError(f"no <plugin> element in {HEADER_FILE.name}")
    last = starts[-1]
    if not lines[last].rstrip().endswith("/>"):  # a plugin with children ends at its closing tag
        closing = [i for i in range(last + 1, len(lines)) if lines[i].strip() == "</plugin>"]
        if not closing:
            raise ValueError(f"the last <plugin> element in {HEADER_FILE.name} is not closed")
        last = closing[0]
    return "\n".join(lines[: last + 1]), "\n".join(lines[last + 1 :])


def _material(key: str) -> str:
    r, g, b = MATERIALS[key]
    colour = f"{r:.2f} {g:.2f} {b:.2f} 1"
    return (
        f"          <material>\n            <ambient>{colour}</ambient>\n"
        f"            <diffuse>{colour}</diffuse>\n            <specular>0.05 0.05 0.05 1</specular>\n"
        f"          </material>\n"
    )


def _geometry(p: Primitive) -> str:
    if p.kind is Kind.POLYGON:
        points = "".join(f"<point>{m(x - p.x)} {m(y - p.y)}</point>" for x, y in p.vertices)
        return f"<polyline>{points}<height>{m(p.height)}</height></polyline>"
    if p.kind is Kind.BOX:
        sx, sy, sz = p.size
        return f"<box><size>{m(sx)} {m(sy)} {m(sz)}</size></box>"
    r, h = p.size
    return f"<cylinder><radius>{m(r)}</radius><length>{m(h)}</length></cylinder>"


def _primitive(p: Primitive) -> str:
    # a box or cylinder is centred on its pose; a polyline is extruded upwards from its pose
    z = p.z_min if p.kind is Kind.POLYGON else p.z_min + p.height / 2
    pose = (
        f"<pose>{m(p.x)} {m(p.y)} {m(z)} 0 0 {rad(p.yaw)}</pose>"  # in the link frame (SDF default)
    )
    geometry = _geometry(p)
    return (
        f'        <collision name="{p.name}_col">\n          {pose}\n'
        f"          <geometry>{geometry}</geometry>\n        </collision>\n"
        f'        <visual name="{p.name}_vis">\n          {pose}\n'
        f"          <geometry>{geometry}</geometry>\n{_material(p.material)}        </visual>\n"
    )


def ledger(world: World, header_sha: str) -> str:
    counts = Counter(p.role.value for p in world.primitives)
    role_line = ", ".join(f"{role}: {n}" for role, n in sorted(counts.items()))
    params = ", ".join(f"{k}={v}" for k, v in sorted(world.parameters.items())) or "none"
    sources = (
        "; ".join(f"{s.get('what', '')} ({s.get('cite', '')})" for s in world.sources) or "none"
    )
    moving_lines = ""
    if world.moving is not None:
        from . import moving  # imported here: moving.py imports this module

        moving_lines = moving.ledger_lines(world.moving)
    return (
        "<!--\n"
        "  Design ledger\n"
        f"  Document: {world.folder}.sdf, SDF 1.9, world `{world.id}` ({world.title}).\n"
        "  Consumer: Gazebo Harmonic (gz-sim 8, sdformat14). Units: SI (metres, radians).\n"
        "  Frames: x east, y north, z up; world origin as in the plan. Every pose is written out and uses SDF's\n"
        "    default frame (model: world; link: model; collision and visual: link). Models and links sit at\n"
        "    their parent origins, so every primitive pose is also its world pose. No relative_to attribute is\n"
        "    used: Gazebo skips serialising models that carry one.\n"
        "  Names: each primitive gives a collision `<name>_col` and a visual `<name>_vis` (unique siblings).\n"
        f"  Generator: worldgen {WORLDGEN_VERSION}.\n"
        f"  Parameters: {params}. Seed: {world.seed if world.seed is not None else 'none'}.\n"
        f"  Sources: {escape(sources)}.\n"
        f"  Header: copied from {HEADER_SOURCE} (SHA-256 {header_sha}).\n"
        f"  Floor: a static box {FLOOR_THICKNESS_M} m thick with its top face at z = 0 (as the Corridor floor).\n"
        f"  Primitives by role: {role_line}. Collision and visual geometry are identical.\n"
        + moving_lines
        + "-->\n"
    )


def write_sdf(world: World) -> str:
    """The complete SDF text of a world: header, ground plane and static structure; for E worlds also the E0
    system plugin (after the standard system plugins) and the moving models (after the structure)."""
    header, header_sha = header_block()
    moving_models = ""
    if world.moving is not None:
        from . import moving  # imported here: moving.py imports this module

        plugins, rest = split_after_plugins(header)
        e0_plugin = moving.sdf_plugin(world.moving).rstrip("\n")
        header = f"{plugins}\n\n{e0_plugin}\n{rest}"
        moving_models = "\n" + moving.sdf_models(world.moving)
    xmin, xmax, ymin, ymax = world.bounds()
    fx = math.ceil((xmax - xmin + 2 * FLOOR_MARGIN_M) * 10) / 10
    fy = math.ceil((ymax - ymin + 2 * FLOOR_MARGIN_M) * 10) / 10
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    link = "structure_link"
    body = "".join(_primitive(p) for p in sorted_primitives(world.static_primitives()))
    world_name = world.folder.lower()
    floor_colour = MATERIALS["floor"]
    floor_rgba = f"{floor_colour[0]:.2f} {floor_colour[1]:.2f} {floor_colour[2]:.2f} 1"
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ledger(world, header_sha)
        + '<sdf version="1.9">\n'
        + f'  <world name="{world_name}">\n'
        + header
        + "\n\n"
        + '    <model name="ground_plane">\n      <static>true</static>\n'
        + "      <pose>0 0 0 0 0 0</pose>\n"
        + '      <link name="ground_link">\n'
        + f"        <pose>{m(cx)} {m(cy)} {m(-FLOOR_THICKNESS_M / 2)} 0 0 0</pose>\n"
        + '        <collision name="ground_collision">\n'
        + f"          <geometry><box><size>{m(fx)} {m(fy)} {m(FLOOR_THICKNESS_M)}</size></box></geometry>\n"
        + "          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>\n"
        + "        </collision>\n"
        + '        <visual name="ground_visual">\n'
        + f"          <geometry><box><size>{m(fx)} {m(fy)} {m(FLOOR_THICKNESS_M)}</size></box></geometry>\n"
        + f"          <material><ambient>{floor_rgba}</ambient><diffuse>{floor_rgba}</diffuse>"
        + "<specular>0.05 0.05 0.05 1</specular></material>\n"
        + "        </visual>\n      </link>\n    </model>\n\n"
        + f'    <model name="{world.structure_name}">\n      <static>true</static>\n'
        + "      <pose>0 0 0 0 0 0</pose>\n"
        + f'      <link name="{link}">\n        <pose>0 0 0 0 0 0</pose>\n'
        + body
        + "      </link>\n    </model>\n"
        + moving_models
        + "  </world>\n</sdf>\n"
    )
