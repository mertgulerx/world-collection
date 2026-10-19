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

"""RP1 ACES3 Austin, rebuilt as a clean designed world from the ACES3 map image.

The layout follows the third floor of the ACES building as mapped by the Radish/Freiburg data set: a corridor ring
around four room blocks, a cross of two corridors, a round rotunda where they meet, two entrance vestibules and a
few alcoves. Positions were measured on the map image at 0.05 m per pixel and rounded to 0.1 m; the SLAM map's
artefacts (beams through doors and glass, drift) are not reproduced. The room blocks the robot never mapped are
solid.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .. import raster
from ..checks import Analysis, no_symmetry
from ..geometry import Recess, grid_walls, walled_rectangle
from ..layout_checks import free_length
from ..families.common import AXIS_RAYS, full_height_raster, longest_ray_heading
from ..world import (
    BoolArray,
    CheckResult,
    Dimension,
    Expected,
    FloatArray,
    Kind,
    Primitive,
    Role,
    Spawn,
    World,
)

UNIT = 0.1
# Inner faces of the outer wall: the outer edges of the corridor ring (plan R3 rev. 2 §R2.2)
X0, X1, Y0, Y1 = -20.3, 24.9, -21.5, 19.2
RING = 1.8  # ring corridor clear width
# free rectangles (x0, x1, y0, y1) inside the outer faces
CORRIDORS = (
    ("ring_west", X0, X0 + RING, Y0, Y1),
    ("ring_east", X1 - RING, X1, Y0, Y1),
    ("ring_south", X0, X1, Y0, Y0 + RING),
    ("ring_north", X0, X1, Y1 - RING, Y1),
    ("cross_ns", -1.6, 1.7, Y0, Y1),  # 3.3 m
    ("cross_ew", X0, X1, -1.4, 1.0),  # 2.4 m
    ("niche_ns", -2.5, -1.6, 5.5, 10.0),  # widening on the west side of the north arm
    ("alcove_ne", 11.0, 14.5, 1.0, 2.9),  # room niche north of the east arm
)
ROTUNDA_R = 3.9  # centred on the crossing (0, 0)
# The rotunda face is a smooth wall of thin rotated boxes whose inner faces are chords of the circle, at most
# ARC_STEP_DEG apart (sagitta under 5 mm). The blocks behind it are carved ARC_CARVE beyond the radius, so their
# 0.1 m steps stay hidden inside the wall (cell corners reach at most r + 0.1 + 0.071 m < r + ARC_T).
ARC_STEP_DEG = 5.0
ARC_T = 0.2
ARC_H = (
    2.02  # 2 cm above the blocks, so the overlapping tops are not coplanar (no flicker in renders)
)
ARC_CARVE = 0.1
# Block corners at the crossing (x bound, y bound) per quadrant: the arms' faces
QUADRANTS = (("ne", 1.7, 1.0), ("nw", -1.6, 1.0), ("sw", -1.6, -1.4), ("se", 1.7, -1.4))
# Entrance vestibules and door recesses (landmarks) as notches in the outer wall: side, id, centre, width, depth
VESTIBULES = (("n", "VN", 0.05, 3.3, 2.5), ("s", "VS", 0.25, 2.9, 2.5))
DOOR_RECESS_W, DOOR_RECESS_D = 1.0, 0.4  # every door recess
# door recess centres along each side of the outer wall
DOORS = {
    "n": (-16.8, -12.5, -3.5, 5.0, 22.0),
    "e": (2.0, -3.0, -6.5, -11.0, -15.0),
    "s": (-12.0, -7.5, 6.0, 11.0, 15.5),
    "w": (9.5, 2.0, -18.5),
}


def _cell_centres() -> tuple[FloatArray, FloatArray]:
    """x and y of the cell centres of the 0.1 m construction grid inside the outer faces."""
    nx, ny = round((X1 - X0) / UNIT), round((Y1 - Y0) / UNIT)
    gx, gy = np.meshgrid(X0 + (np.arange(nx) + 0.5) * UNIT, Y0 + (np.arange(ny) + 0.5) * UNIT)
    return gx, gy


def _corridor_mask() -> BoolArray:
    """Corridor and niche cells only (no rotunda): exact on the 0.1 m grid, since every edge lies on it."""
    gx, gy = _cell_centres()
    free = np.zeros(gx.shape, dtype=bool)
    for _, a, b, c, d in CORRIDORS:
        free |= (gx >= a) & (gx <= b) & (gy >= c) & (gy <= d)
    return free


def _free_mask() -> tuple[BoolArray, int, int]:
    """Free cells on the construction grid (cell centres tested): the corridors and the rotunda disk carved
    ARC_CARVE beyond its radius; also the grid's column and row counts."""
    gx, gy = _cell_centres()
    carve = ROTUNDA_R + ARC_CARVE
    free = _corridor_mask() | (gx * gx + gy * gy <= carve * carve)
    ny, nx = free.shape
    return free, nx, ny


def _arc(xb: float, yb: float) -> list[tuple[float, float]]:
    """Points on the rotunda circle from the y = yb face to the x = xb face of one block, ≤ ARC_STEP_DEG apart."""
    r = ROTUNDA_R
    a0 = math.atan2(yb, math.copysign(math.sqrt(r * r - yb * yb), xb))
    a1 = math.atan2(math.copysign(math.sqrt(r * r - xb * xb), yb), xb)
    n = math.ceil(abs(a1 - a0) / math.radians(ARC_STEP_DEG))
    return [
        (r * math.cos(a0 + (a1 - a0) * k / n), r * math.sin(a0 + (a1 - a0) * k / n))
        for k in range(n + 1)
    ]


def _rotunda_wall() -> tuple[list[Primitive], float]:
    """The smooth rotunda face and the free area it adds to the corridors (shoelace over each quadrant)."""
    prims: list[Primitive] = []
    added = 0.0
    for name, xb, yb in QUADRANTS:
        pts = _arc(xb, yb)
        last = len(pts) - 2
        for k, ((ax, ay), (bx, by)) in enumerate(zip(pts, pts[1:], strict=False)):
            length = math.hypot(bx - ax, by - ay)
            dx, dy = (bx - ax) / length, (by - ay) / length
            # Neighbouring boxes splay apart outside each joint; lengthen them there so they overlap. The
            # extension lies outside the free polygon; the arc's two ends meet the arms' faces and stay flush.
            half = math.asin(length / (2 * ROTUNDA_R))  # half the angle between neighbouring chords
            ext = ARC_T * math.tan(half) + 0.01
            ea, eb = (ext if k > 0 else 0.0), (ext if k < last else 0.0)
            ax, ay, bx, by = ax - dx * ea, ay - dy * ea, bx + dx * eb, by + dy * eb
            mx, my = (ax + bx) / 2, (ay + by) / 2
            ux, uy = dy, -dx  # chord normal; pick the outward one
            if ux * mx + uy * my < 0:
                ux, uy = -ux, -uy
            prims.append(
                Primitive(
                    Kind.BOX,
                    f"rotunda_{name}_{k:02d}",
                    round(mx + ux * ARC_T / 2, 4),
                    round(my + uy * ARC_T / 2, 4),
                    round(math.atan2(dy, dx), 6),
                    (round(length + ea + eb, 4), ARC_T, ARC_H),
                    Role.WALL,
                    "core",
                )
            )
        poly = [(xb, yb), *pts]
        added += (
            abs(
                sum(
                    p[0] * q[1] - q[0] * p[1]
                    for p, q in zip(poly, poly[1:] + poly[:1], strict=True)
                )
            )
            / 2
        )
    return prims, added


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    results = [no_symmetry("RP1 layout has no symmetry", world)]
    widths = {
        "ring_west": free_length(a, "x", -10.0, X0 - 0.5, X0 + RING + 0.5),
        "ring_north": free_length(a, "y", 8.0, Y1 - RING - 0.5, Y1 + 0.5),
        "cross_ns": free_length(a, "x", 12.0, -3.0, 3.0),
        "cross_ew": free_length(a, "y", -10.0, -3.0, 3.0),
    }
    want = {"ring_west": RING, "ring_north": RING, "cross_ns": 3.3, "cross_ew": 2.4}
    results.append(
        CheckResult(
            "RP1 corridor widths: ring 1.8 m, north–south arm 3.3 m, east–west arm 2.4 m",
            all(abs(widths[k] - want[k]) <= a.grid.h + 1e-9 for k in want),
            widths,
        )
    )
    across = (round(X1 - X0 - RING, 2), round(Y1 - Y0 - RING, 2))
    results.append(
        CheckResult(
            "RP1 ring centrelines about 40 m across (36–46 m; the source states ~40 m)",
            all(36.0 <= v <= 46.0 for v in across),
            {"x_m": across[0], "y_m": across[1]},
        )
    )
    r = a.grid.cell(ROTUNDA_R - 0.1, 0.0)
    results.append(
        CheckResult(
            "RP1 rotunda free to its radius",
            bool(a.reachable[r]) and bool(a.reachable[a.grid.cell(0.0, ROTUNDA_R - 0.1)]),
            ROTUNDA_R,
        )
    )
    return results


def _far_spawn(prims: tuple[Primitive, ...], start: tuple[float, float]) -> tuple[float, float]:
    """S2: the free cell with clearance ≥ 0.7 m that is geodesically farthest from S1 (plan §8)."""
    xs = [p.footprint_bounds() for p in prims]
    grid = raster.Grid.around(
        (min(b[0] for b in xs), max(b[1] for b in xs), min(b[2] for b in xs), max(b[3] for b in xs))
    )
    occ = raster.rasterise(grid, prims)
    reach = raster.component_at(~occ, grid.cell(*start))
    dist = raster.clearance(reach, grid.h)
    ok = reach & (dist >= 0.7)
    cells = [(int(c[0]), int(c[1])) for c in np.argwhere(ok)]
    lengths = raster.geodesic(reach, grid.cell(*start), cells, grid.h)
    far = cells[int(np.argmax(lengths))]
    x, y = grid.centre(*far)
    return round(x, 2), round(y, 2)


def build() -> World:
    free, nx, ny = _free_mask()
    recesses: dict[str, list[Recess]] = {side: [] for side in ("n", "e", "s", "w")}
    for side, rid, centre, width, depth in VESTIBULES:
        recesses[side].append(Recess(rid, centre, width, depth))
    for side, centres in DOORS.items():
        for i, c in enumerate(centres):
            recesses[side].append(
                Recess(f"D{side.upper()}{i + 1}", c, DOOR_RECESS_W, DOOR_RECESS_D)
            )
    sides: dict[str, Iterable[Recess]] = dict(recesses)
    outer, notches = walled_rectangle(X0, X1, Y0, Y1, recesses=sides)
    blocks = grid_walls(
        ~free, UNIT, origin=(X0, Y0), prefix="block", role=Role.WALL, material="core"
    )
    arc, rotunda_area = _rotunda_wall()
    prims = (*outer, *blocks, *arc)
    area = float(np.count_nonzero(_corridor_mask())) * UNIT * UNIT + rotunda_area
    doors = sum(len(v) for v in DOORS.values())
    area += sum(w * d for _, _, _, w, d in VESTIBULES) + DOOR_RECESS_W * DOOR_RECESS_D * doors
    area = round(area, 4)
    s2 = _far_spawn(prims, (0.0, 0.0))
    s2_yaw = longest_ray_heading(*full_height_raster(prims), s2[0], s2[1], AXIS_RAYS)
    return World(
        id="RP1",
        slug="aces3_austin",
        title="ACES3 Austin (rebuilt from the map image)",
        tier="RP",
        role="evaluation",
        primitives=prims,
        spawns=(
            Spawn(
                "S1",
                0.0,
                0.0,
                math.pi / 2,
                "rotunda centre, the crossing of the cross, facing north",
            ),
            Spawn(
                "S2",
                s2[0],
                s2[1],
                s2_yaw,
                "farthest free cell from S1 (geodesic), clearance ≥ 0.7 m; longest axis ray",
            ),
            Spawn(
                "S3",
                round(X1 - RING / 2, 2),
                -0.2,
                math.pi,
                "east T junction of the cross and the ring, facing west",
            ),
        ),
        expected=Expected(
            free_area_m2=round(area, 2), narrowest_passage_m=RING, loops=4, loops_unsized=4
        ),
        notches=tuple(notches),
        dimensions=(
            Dimension((X0, Y0 - 3.2), (X1, Y0 - 3.2), f"{X1 - X0:.1f} m"),
            Dimension((X0 - 1.2, Y0), (X0 - 1.2, Y1), f"{Y1 - Y0:.1f} m"),
            Dimension((-1.6, -8.0), (1.7, -8.0), "3.3 m"),
        ),
        sources=(
            {
                "what": "layout of the ACES3 floor (map image; rebuilt, not converted)",
                "cite": "howard2003radish",
            },
            {"what": "ACES3 data set in the SLAM benchmark", "cite": "kummerle2009slam"},
        ),
        extra_checks=(_checks,),
        notice=(
            "RP1 ACES3 Austin is a clean rebuild, in our own geometry, of the layout of the third floor of the ACES\n"
            "building, University of Texas at Austin, as mapped in the data set ut_austin_aces3 of the Robotics\n"
            "Data Set Repository (Radish; A. Howard and N. Roy, 2003), provided by Patrick Beeson, and published\n"
            "as a map image by Cyrill Stachniss. No data from the source is included: positions were measured on\n"
            "the map image (0.05 m per pixel) and rounded to 0.1 m.\n"
        ),
        plan_data={
            "scale_m_per_px": 0.05,
            "ring_centrelines_m": [round(X1 - X0 - RING, 2), round(Y1 - Y0 - RING, 2)],
            "construction_free_area_m2": round(area, 4),
            "grid": [nx, ny],
        },
    )
