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

"""Review renders, run inside Blender (T0 §8).

blender --background --factory-startup --python-exit-code 1 --python worldgen/blender_scene.py -- \
    --world <folder>/review/world.json --out <folder>/review

Builds the scene with bpy.data meshes (primitives sharing a material share one mesh), Principled BSDF
materials, a sun and a world colour, and renders with Cycles on the CPU: an oblique overview, an orthographic
top view and, only when requested with `--only eye_S<n>`, an eye-level view per spawn (0.3 m above the floor,
90° horizontal field of view).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SAMPLES = 32
RES_X, RES_Y = 1280, 800
CYL_SEGMENTS = 32


def box_mesh(p, verts, faces):
    sx, sy, sz = p["size"]
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    base = len(verts)
    for z in (p["z_min"], p["z_min"] + sz):
        for u, v in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            verts.append((p["x"] + u * c - v * s, p["y"] + u * s + v * c, z))
    b = base
    faces += [
        (b, b + 3, b + 2, b + 1),
        (b + 4, b + 5, b + 6, b + 7),
        (b, b + 1, b + 5, b + 4),
        (b + 1, b + 2, b + 6, b + 5),
        (b + 2, b + 3, b + 7, b + 6),
        (b + 3, b, b + 4, b + 7),
    ]


def cylinder_mesh(p, verts, faces):
    r, h = p["size"]
    base = len(verts)
    for z in (p["z_min"], p["z_min"] + h):
        for k in range(CYL_SEGMENTS):
            a = 2 * math.pi * k / CYL_SEGMENTS
            verts.append((p["x"] + r * math.cos(a), p["y"] + r * math.sin(a), z))
    n = CYL_SEGMENTS
    faces.append(tuple(base + k for k in reversed(range(n))))
    faces.append(tuple(base + n + k for k in range(n)))
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((base + k, base + k2, base + n + k2, base + n + k))


def polygon_mesh(p, verts, faces):
    """A prism over the counter-clockwise polygon `vertices`: bottom, top and one quad per side."""
    pts = p["vertices"]
    n = len(pts)
    base = len(verts)
    for z in (p["z_min"], p["z_min"] + p["size"][0]):
        for x, y in pts:
            verts.append((x, y, z))
    faces.append(tuple(base + k for k in reversed(range(n))))
    faces.append(tuple(base + n + k for k in range(n)))
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((base + k, base + k2, base + n + k2, base + n + k))


MESHES = {"box": box_mesh, "cylinder": cylinder_mesh, "polygon": polygon_mesh}


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    return mat


def add_mesh(name, verts, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def camera(name, location, target=None, rotation=None, ortho_scale=None, hfov_deg=None):
    cam = bpy.data.cameras.new(name)
    if ortho_scale is not None:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho_scale
    if hfov_deg is not None:
        cam.sensor_fit = "HORIZONTAL"
        cam.angle = math.radians(hfov_deg)
    cam.clip_start, cam.clip_end = 0.05, 5000.0
    obj = bpy.data.objects.new(name, cam)
    obj.location = location
    if target is not None:
        direction = Vector(target) - Vector(location)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if rotation is not None:
        obj.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(obj)
    return obj


def render(cam, path):
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.world).read_text(encoding="utf-8"))
    out = Path(args.out)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = 0
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RES_X, RES_Y
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.3

    mats = {k: material(k, v) for k, v in data["materials"].items()}
    groups = {}
    for p in data["primitives"]:
        if p["role"] == "visual":
            continue
        verts, faces = groups.setdefault(p["material"], ([], []))
        MESHES[p["kind"]](p, verts, faces)
    for key, (verts, faces) in sorted(groups.items()):
        add_mesh(f"mesh_{key}", verts, faces, mats[key])

    xmin, xmax, ymin, ymax = data["bounds"]
    margin = data["floor_margin_m"]
    fx0, fx1, fy0, fy1 = xmin - margin, xmax + margin, ymin - margin, ymax + margin
    add_mesh(
        "floor",
        [(fx0, fy0, 0), (fx1, fy0, 0), (fx1, fy1, 0), (fx0, fy1, 0)],
        [(0, 1, 2, 3)],
        mats["floor"],
    )
    for n in data["notches"]:
        # a thin blue mark on the floor of each free-space landmark notch
        verts = [(x, y, 0.005) for x, y in n["polygon"]]
        add_mesh(f"notch_{n['id']}", verts, [tuple(range(len(verts)))], mats["landmark"])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.2
    sun.angle = math.radians(8)
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
    scene.collection.objects.link(sun_obj)
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.74, 0.90, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    scene.world = world

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin)
    wanted = set(filter(None, args.only.split(",")))

    def want(name):
        return not wanted or name in wanted

    if want("overview"):
        elev = math.radians(data.get("overview_elevation_deg", 45.0))
        # framed for 45°; a steeper camera sees the ground's depth more fully, so it backs off in proportion
        dist = (
            0.95 * span / math.tan(math.radians(22)) * math.sin(elev) / math.sin(math.radians(45))
        )
        azim = math.radians(-120)
        loc = (
            cx + dist * math.cos(elev) * math.cos(azim),
            cy + dist * math.cos(elev) * math.sin(azim),
            dist * math.sin(elev),
        )
        cam = camera("overview", loc, target=(cx, cy, 0.0), hfov_deg=40)
        render(cam, out / "render_overview.png")
    if want("top"):
        scale = 1.05 * max(xmax - xmin, (ymax - ymin) * RES_X / RES_Y)
        cam = camera("top", (cx, cy, 200.0), rotation=(0.0, 0.0, 0.0), ortho_scale=scale)
        render(cam, out / "render_top.png")
    for s in data["spawns"]:
        if f"eye_{s['id']}" in wanted:  # eye views only on request (maintainer, 18 October 2026)
            cam = camera(
                f"eye_{s['id']}",
                (s["x"], s["y"], 0.3),
                rotation=(math.pi / 2, 0.0, s["yaw"] - math.pi / 2),
                hfov_deg=90,
            )
            render(cam, out / f"render_eye_{s['id']}.png")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
