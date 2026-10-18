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

"""SC5 Nested pockets (plan docs/plans/worlds/B5-nested-pockets.md).

A start hall with two kinds of exits: 1.2 m doors straight into three large rooms, and 4 m openings into three
chains of five nested pockets (24, 15, 12, 8 and 6 m² by design), each entered through a side or corner opening
so that the inner pockets cannot be seen from the hall. Frontier size and the area behind it disagree on purpose.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..checks import Analysis, no_symmetry
from ..families.common import AXIS_RAYS, full_height_raster, longest_ray_heading
from ..geometry import door, grid_walls
from ..layout_checks import door_openings, rect_area
from ..world import BoolArray, CheckResult, Dimension, Expected, Role, Spawn, World

UNIT = 0.1  # construction grid (m); every coordinate below is a multiple of it
HALF_WALL = 0.1  # walls are 0.2 m thick, centred on the region outlines below
DOOR_M = 1.2  # designed-world door width (B4)
OPENING_M = 4.0  # the wide openings from the hall into the pocket chains
LIDAR_M = 10.0  # MOGI range, for the visibility checks

# Region outlines on the wall centre lines: id -> (x0, y0, x1, y1). The interior is inset by HALF_WALL.
HALL = (-7.0, -5.0, 7.0, 5.0)
ROOMS = {
    "R1": (2.0, 5.0, 16.0, 20.6),  # north-east
    "R2": (-23.8, -5.0, -7.0, 5.0),  # west
    "R3": (2.0, -20.6, 16.0, -5.0),  # south-east
}
# Pocket chains, outer to innermost. North and south mirror each other except the shape of the last two pockets,
# which breaks the symmetry; the east chain points east between R1 and R3.
POCKETS = {
    "N1": (-7.0, 5.0, 1.0, 8.0),
    "N2": (-10.0, 5.6, -7.0, 11.4),
    "N3": (-7.0, 9.0, -3.0, 12.0),
    "N4": (-8.3, 12.0, -3.0, 13.5),
    "N5": (-9.1, 13.5, -6.6, 15.9),
    "S1": (-7.0, -8.0, 1.0, -5.0),
    "S2": (-10.0, -11.4, -7.0, -5.6),
    "S3": (-7.0, -12.0, -3.0, -9.0),
    "S4": (-8.0, -13.6, -3.0, -12.0),
    "S5": (-8.8, -16.0, -6.3, -13.6),
    "E1": (7.0, -2.1, 12.8, 2.1),
    "E2": (9.0, 2.1, 14.0, 5.0),
    "E3": (14.0, 1.0, 17.0, 5.0),
    "E4": (17.0, 1.0, 18.5, 6.3),
    "E5": (18.5, 4.6, 21.0, 7.0),
}
CHAINS = ("N", "S", "E")
LEVELS = 5  # pockets per chain
# Openings on a shared outline: id -> (axis of the wall, position of the wall, start, end along the wall).
OPENINGS = {
    "D_R1": ("x", 5.0, 3.4, 4.6),
    "D_R2": ("y", -7.0, -0.6, 0.6),
    "D_R3": ("x", -5.0, 3.4, 4.6),
    "O_N1": ("x", 5.0, -3.5, 0.5),
    "O_N2": ("y", -7.0, 6.1, 7.9),
    "O_N3": ("y", -7.0, 9.1, 10.9),
    "O_N4": ("x", 12.0, -5.1, -3.1),
    "O_N5": ("x", 13.5, -8.2, -6.7),
    "O_S1": ("x", -5.0, -3.5, 0.5),
    "O_S2": ("y", -7.0, -7.9, -6.1),
    "O_S3": ("y", -7.0, -10.9, -9.1),
    "O_S4": ("x", -12.0, -5.1, -3.1),
    "O_S5": ("x", -13.6, -7.9, -6.4),
    "O_E1": ("y", 7.0, -2.0, 2.0),
    "O_E2": ("x", 2.1, 11.0, 12.7),
    "O_E3": ("y", 14.0, 3.0, 4.5),
    "O_E4": ("y", 17.0, 1.1, 3.1),
    "O_E5": ("y", 18.5, 4.7, 6.2),
}
DOORS = tuple(
    door(
        oid,
        (a0 + a1) / 2 if axis == "x" else at,
        at if axis == "x" else (a0 + a1) / 2,
        round(a1 - a0, 3),
        axis,
    )
    for oid, (axis, at, a0, a1) in OPENINGS.items()
)
REGIONS = {"H": HALL, **ROOMS, **POCKETS}
SPAWNS = (
    ("S1", 0.0, 0.0, "hall centre, every exit 5–7 m away"),
    ("S2", 5.5, 3.5, "north-east corner of the hall, next to the R1 door"),
    ("S3", -5.5, -3.5, "south-west corner of the hall, next to the south pocket opening"),
)


def _dm(v: float) -> int:
    """A coordinate in construction cells (decimetres)."""
    return round(v / UNIT)


def _layout() -> tuple[BoolArray, tuple[float, float]]:
    """The construction-grid occupancy (True = wall) and its origin. Free space is the inset interiors plus the
    openings; walls are a 0.2 m band around it, and pockets of space enclosed by walls are filled solid."""
    xs = [v for r in REGIONS.values() for v in (r[0], r[2])]
    ys = [v for r in REGIONS.values() for v in (r[1], r[3])]
    cx0, cy0 = _dm(min(xs)) - 3, _dm(min(ys)) - 3
    free = np.zeros((_dm(max(ys)) + 3 - cy0, _dm(max(xs)) + 3 - cx0), dtype=bool)
    inset = _dm(HALF_WALL)
    for x0, y0, x1, y1 in REGIONS.values():
        free[
            _dm(y0) + inset - cy0 : _dm(y1) - inset - cy0,
            _dm(x0) + inset - cx0 : _dm(x1) - inset - cx0,
        ] = True
    for axis, at, a0, a1 in OPENINGS.values():
        if axis == "x":
            free[_dm(at) - inset - cy0 : _dm(at) + inset - cy0, _dm(a0) - cx0 : _dm(a1) - cx0] = (
                True
            )
        else:
            free[_dm(a0) - cy0 : _dm(a1) - cy0, _dm(at) - inset - cx0 : _dm(at) + inset - cx0] = (
                True
            )
    band = ndimage.binary_dilation(
        free, structure=np.ones((3, 3), dtype=bool), iterations=2 * inset
    )
    walls = ndimage.binary_fill_holes(band) & ~free
    return walls, (cx0 * UNIT, cy0 * UNIT)


def _interior(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return x0 + HALF_WALL, x1 - HALF_WALL, y0 + HALF_WALL, y1 - HALF_WALL


def _seen(a: Analysis, points: list[tuple[float, float]]) -> BoolArray:
    """Free cells a 2D lidar of range LIDAR_M sees from any of the points (rays stop at the first non-free cell)."""
    g = a.grid
    free = a.reachable
    n_rays = int(math.ceil(2 * math.pi * LIDAR_M / (g.h / 2)))
    angles = np.arange(n_rays, dtype=np.float64) * (2 * math.pi / n_rays)
    steps = np.arange(0, int(math.ceil(LIDAR_M / (g.h / 2))) + 1, dtype=np.float64) * (g.h / 2)
    seen = np.zeros_like(free)
    for x, y in points:
        r0, c0 = g.cell(x, y)
        rr = np.floor(r0 + 0.5 + np.outer(np.sin(angles), steps) / g.h).astype(np.int64)
        cc = np.floor(c0 + 0.5 + np.outer(np.cos(angles), steps) / g.h).astype(np.int64)
        inside = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
        ok = np.zeros_like(inside)
        ok[inside] = free[rr[inside], cc[inside]]
        reach = np.cumprod(ok, axis=1).astype(bool)
        seen[rr[reach], cc[reach]] = True
    return seen


def _share(a: Analysis, seen: BoolArray, rect: tuple[float, float, float, float]) -> float:
    """Share of a region's reachable interior that is in `seen`."""
    g = a.grid
    x0, x1, y0, y1 = _interior(rect)
    r0, c0 = g.cell(x0 + g.h / 2, y0 + g.h / 2)
    r1, c1 = g.cell(x1 - g.h / 2, y1 - g.h / 2)
    region = a.reachable[r0 : r1 + 1, c0 : c1 + 1]
    return round(
        float(np.count_nonzero(seen[r0 : r1 + 1, c0 : c1 + 1] & region))
        / float(np.count_nonzero(region)),
        3,
    )


def _grid_points(rect: tuple[float, float, float, float], step: float) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = _interior(rect)
    return [
        (round(float(x), 3), round(float(y), 3))
        for x in np.arange(x0 + 0.4, x1 - 0.3, step)
        for y in np.arange(y0 + 0.4, y1 - 0.3, step)
    ]


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    results = [no_symmetry("SC5 layout has no symmetry", world)]
    widths, separated = door_openings(a, DOORS)
    widths_ok = [abs(w - d.width) < a.grid.h / 2 for w, d in zip(widths, DOORS, strict=True)]
    results.append(
        CheckResult(
            "SC5 every door and opening has its listed width",
            all(widths_ok),
            dict(zip([d.id for d in DOORS], widths, strict=True)),
        )
    )
    results.append(
        CheckResult(
            "SC5 cutting each door or opening separates exactly what lies behind it",
            all(separated),
            separated.count(True),
        )
    )
    areas = {rid: rect_area(a, *_interior(r)) for rid, r in REGIONS.items()}
    rooms = sum(areas[r] for r in ROOMS)
    pockets = sum(areas[p] for p in POCKETS)
    nested = all(
        all(areas[f"{c}{i}"] > areas[f"{c}{i + 1}"] for i in range(1, LEVELS)) for c in CHAINS
    )
    results.append(CheckResult("SC5 every pocket chain shrinks inwards", nested, areas))
    results.append(
        CheckResult(
            "SC5 the rooms hold more area than the pocket chains (≥ 3×)",
            rooms >= 3 * pockets,
            {"rooms_m2": round(rooms, 2), "pockets_m2": round(pockets, 2)},
        )
    )
    hall_seen = _seen(a, _grid_points(HALL, 1.0))
    from_hall = {rid: _share(a, hall_seen, r) for rid, r in REGIONS.items() if rid != "H"}
    hidden = all(from_hall[f"{c}{i}"] == 0.0 for c in CHAINS for i in (LEVELS - 1, LEVELS))
    results.append(
        CheckResult(
            "SC5 the two innermost pockets are not visible from the hall (10 m lidar)",
            hidden,
            from_hall,
        )
    )
    # Level 4 is entered at a corner of level 3; level 5 is a square at the far end of level 4, out of sight
    # until the robot is inside level 4.
    from_level3 = {c: _seen(a, _grid_points(POCKETS[f"{c}{LEVELS - 2}"], 0.5)) for c in CHAINS}
    corner = {
        f"{c}{i}": _share(a, from_level3[c], POCKETS[f"{c}{i}"])
        for c in CHAINS
        for i in (LEVELS - 1, LEVELS)
    }
    results.append(
        CheckResult(
            "SC5 from the third pocket, the fourth is partly hidden (< 90 %) and the fifth mostly (< 50 %)",
            all(corner[f"{c}{LEVELS - 1}"] < 0.9 and corner[f"{c}{LEVELS}"] < 0.5 for c in CHAINS),
            corner,
        )
    )
    return results


def build() -> World:
    walls, origin = _layout()
    prims = tuple(
        grid_walls(walls, UNIT, origin=origin, prefix="wall", role=Role.WALL, material="inner")
    )
    grid, occ = full_height_raster(prims)
    spawns = tuple(
        Spawn(
            sid, x, y, longest_ray_heading(grid, occ, x, y, AXIS_RAYS), f"{note}; longest axis ray"
        )
        for sid, x, y, note in SPAWNS
    )
    return World(
        id="SC5",
        slug="nested_pockets",
        title="Nested pockets",
        tier="SC",
        role="evaluation",
        plan="docs/plans/worlds/B5-nested-pockets.md",
        primitives=prims,
        spawns=spawns,
        expected=Expected(free_area_m2=907.74, narrowest_passage_m=1.2, loops=0, loops_unsized=0),
        doors=DOORS,
        dimensions=(
            Dimension((3.4, 4.4), (4.6, 4.4), "door 1.2 m"),
            Dimension((-3.5, 4.4), (0.5, 4.4), "opening 4.0 m"),
        ),
        sources=(
            {
                "what": "layout idea (frontier utility, missed small areas)",
                "cite": "xu2022explorebench",
            },
        ),
        extra_checks=(_checks,),
    )
