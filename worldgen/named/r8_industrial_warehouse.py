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

"""RP2 Industrial warehouse (plan docs/plans/worlds/R8-industrial-warehouse.md); the base world of E3.

Every element is a primitive defined here (plan §2: our own aisles, no third-party file). All boxes lie on the
0.05 m grid, so the raster is exact (plan §3).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, spatial

from ..checks import Analysis, no_symmetry
from ..geometry import Recess, box, door, walled_rectangle
from ..layout_checks import door_cut, door_openings, free_length, rect_area
from ..world import CheckResult, Dimension, Expected, Kind, Point, Role, Spawn, World

PLAN = "docs/plans/worlds/R8-industrial-warehouse.md"
DOOR_M = 1.2  # designed-world door width (maintainer, 18 October 2026)
HALL = (-28.0, 28.0, -18.0, 18.0)  # inner faces of the outer wall (plan §3.1)
OUTER_T, OUTER_H = 0.3, 4.0
# dock door id, centre x; each 3.0 m wide and 0.6 m deep in the south wall (plan §3.1)
DOCKS = (("DOCK1", -20.0), ("DOCK2", -13.0), ("DOCK3", -5.0), ("DOCK4", 4.0))
DOCK_W, DOCK_D = 3.0, 0.6
RACK_H = 3.0
# rack id, x range, y range (plan §3.2)
RACKS = (
    ("rack_r1_w", -26.0, -9.0, -8.0, -5.8),
    ("rack_r1_e", -5.0, 12.0, -8.0, -5.8),
    ("rack_r2_w", -26.0, -9.0, -2.8, -0.6),
    ("rack_r2_e", -5.0, 12.0, -2.8, -0.6),
    ("rack_r3_w", -23.0, -9.0, 2.4, 4.6),
    ("rack_r3_e", -5.0, 12.0, 2.4, 4.6),
    ("rack_r4_w", -26.0, -9.0, 7.1, 9.3),
    ("rack_r4_e", -5.0, 12.0, 7.1, 9.3),
    ("rack_r5", -26.0, 12.0, 16.9, 18.0),
)
COLUMN_HALF, COLUMN_H = 0.2, 4.0
# column id, centre (plan §3.4)
COLUMNS = (
    ("col_01", -21.0, -13.0),
    ("col_02", -7.0, -13.0),
    ("col_03", 7.0, -13.0),
    ("col_04", 21.0, -13.0),
    ("col_05", 21.0, 0.0),
    ("col_06", -21.0, 16.7),
    ("col_07", -7.0, 16.7),
    ("col_08", 7.0, 16.7),
)
OFFICE_WALL_H = 2.5
# office wall id, x range, y range (plan §3.5)
OFFICE_WALLS = (
    ("office_w", 17.9, 18.1, 9.9, 18.0),
    ("office_s_1", 18.1, 19.4, 9.9, 10.1),
    ("office_s_2", 20.6, 28.0, 9.9, 10.1),
    ("office_p_1", 22.9, 23.1, 10.1, 11.4),
    ("office_p_2", 22.9, 23.1, 12.6, 18.0),
)
DOORS = (
    door("D01", 20.0, 10.0, DOOR_M, "x"),  # office door, south wall
    door("D02", 23.0, 12.0, DOOR_M, "y"),  # partition door
)
# room id, interior x range, interior y range, a point inside (plan §3.5)
OFFICE_ROOMS = (
    ("west", (18.1, 22.9), (10.1, 18.0), (21.0, 14.0)),
    ("east", (23.1, 28.0), (10.1, 18.0), (25.5, 14.0)),
)
# id, x range, y range, height, role, material (plan §3.5 cabinet and §3.6)
LOADS = (
    ("office_cabinet", 19.7, 21.3, 17.5, 18.0, 2.0, Role.FURNITURE, "furniture"),
    ("pallet_01", -17.6, -16.4, -13.5, -12.5, 1.2, Role.OBSTACLE, "inner"),
    ("pallet_02", -16.4, -15.2, -13.5, -12.5, 1.2, Role.OBSTACLE, "inner"),
    ("pallet_03", -2.6, -1.4, -13.5, -12.5, 1.2, Role.OBSTACLE, "inner"),
    ("block_01", 16.4, 17.6, 2.5, 3.5, 1.5, Role.OBSTACLE, "inner"),
    ("block_02", 17.6, 18.8, 2.5, 3.5, 1.5, Role.OBSTACLE, "inner"),
    ("block_03", 16.4, 17.6, 3.5, 4.5, 1.5, Role.OBSTACLE, "inner"),
    ("charger_01", 27.6, 28.0, -4.3, -3.7, 1.2, Role.LANDMARK, "landmark"),
    ("charger_02", 27.6, 28.0, -2.5, -1.9, 1.2, Role.LANDMARK, "landmark"),
    ("charger_03", 27.6, 28.0, -0.7, -0.1, 1.2, Role.LANDMARK, "landmark"),
    ("lm_cabinet_w", -28.0, -27.4, 4.4, 5.6, 2.0, Role.LANDMARK, "landmark"),
)


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> list[Point]:
    """A closed counter-clockwise rectangle polyline starting at its south-west corner."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


# path id, description, polylines (plan §3.7); E3 reads them from world.yaml `plan_data.patrol_paths`
PATROL_PATHS: tuple[tuple[str, str, tuple[tuple[Point, ...], ...]], ...] = (
    ("P-A", "staging loop", (tuple(_rectangle(-24.0, 10.0, -15.5, -10.5)),)),
    (
        "P-B",
        "aisle runs A1, A2, A4",
        (
            ((-25.0, -4.3), (12.0, -4.3)),
            ((-25.0, 0.9), (12.0, 0.9)),
            ((-25.0, 10.8), (12.0, 10.8)),
        ),
    ),
    ("P-C", "cross run", (((-7.0, -9.5), (-7.0, 14.0)),)),
    ("P-D", "north run", (((-26.0, 14.0), (12.5, 14.0)),)),
    ("P-E", "east loop", (tuple(_rectangle(14.0, 24.0, -6.0, 7.0)),)),
)
PATH_CLEARANCE_M = 1.5  # static geometry to every path line (plan §3.7)
SPAWN_PATH_M = 2.5  # spawn to every path line (plan §5)
SPAWN_CLEARANCE_M = 1.5  # spawn to walls and obstacles (plan §5)
SLOT_M = 1.0  # no slot narrower than this between boxes (plan §7)
SAMPLE_M = 0.05
# aisle (plan §3.3), axis of the measuring line, position, range, expected free width
AISLES = (
    ("A1", "y", -17.5, -5.8, -2.8, 3.0),
    ("A2", "y", -17.5, -0.6, 2.4, 3.0),
    ("A3", "y", -17.5, 4.6, 7.1, 2.5),
    ("A4 + north aisle, R4 to R5", "y", -17.5, 9.3, 16.9, 7.6),
    ("A4 + north aisle at col_06", "y", -21.0, 9.3, 16.9, 7.2),
    ("cross aisle", "x", -4.3, -9.0, -5.0, 4.0),
    ("west aisle", "x", -4.3, -28.0, -26.0, 2.0),
    ("west aisle at lm_cabinet_w", "x", 5.0, -28.0, -26.0, 1.4),
    ("R3 bay with west aisle", "x", 3.5, -28.0, -23.0, 5.0),
)


def _samples(polyline: tuple[Point, ...]) -> list[Point]:
    """Points every 0.05 m along a polyline, both ends included."""
    pts: list[Point] = []
    for (x0, y0), (x1, y1) in zip(polyline[:-1], polyline[1:], strict=True):
        n = max(1, round(math.hypot(x1 - x0, y1 - y0) / SAMPLE_M))
        pts += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    pts.append(polyline[-1])
    return pts


def _segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _path_distance(p: Point, polylines: tuple[tuple[Point, ...], ...]) -> float:
    return min(
        _segment_distance(p, a, b)
        for line in polylines
        for a, b in zip(line[:-1], line[1:], strict=True)
    )


def _box_gap(
    a: tuple[float, ...], b: tuple[float, ...]
) -> tuple[float, tuple[float, float, float, float]]:
    """Euclidean gap between two axis-aligned footprints (xmin, xmax, ymin, ymax), 0 when they touch, and the
    rectangle between them (per axis: the space between the footprints, or their common range)."""
    lo_x, hi_x = sorted((max(a[0], b[0]), min(a[1], b[1])))
    lo_y, hi_y = sorted((max(a[2], b[2]), min(a[3], b[3])))
    dx = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    dy = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    return math.hypot(dx, dy), (lo_x, hi_x, lo_y, hi_y)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    g = a.grid
    results = [no_symmetry("RP2 dock, rack, column and office layout has no symmetry", world)]

    # plan §7: no primitive within 1.5 m of a patrol path, on the raster. The distance runs from each path
    # sample to the nearest occupied cell square (exact for faces on the 0.05 m grid).
    rows, cols = np.nonzero(a.occupied)
    centres = np.column_stack([g.x0 + (cols + 0.5) * g.h, g.y0 + (rows + 0.5) * g.h])
    tree = spatial.cKDTree(centres)
    path_min: dict[str, list[float]] = {}
    for pid, _, polylines in PATROL_PATHS:
        pts = np.array([p for line in polylines for p in _samples(line)])
        _, idx = tree.query(pts, k=8)
        near = centres[idx]
        gap = np.maximum(np.abs(near - pts[:, None, :]) - g.h / 2, 0.0)
        dist = np.hypot(gap[..., 0], gap[..., 1]).min(axis=1)
        i = int(np.argmin(dist))
        path_min[pid] = [
            round(float(dist[i]), 3),
            round(float(pts[i, 0]), 2),
            round(float(pts[i, 1]), 2),
        ]
    results.append(
        CheckResult(
            "RP2 no static geometry within 1.5 m of a patrol path (raster)",
            all(v[0] >= PATH_CLEARANCE_M - 1e-6 for v in path_min.values()),
            path_min,
            "per path: minimum distance (m) and the path point where it occurs",
        )
    )

    spawn_path = {
        s.id: round(min(_path_distance((s.x, s.y), lines) for _, _, lines in PATROL_PATHS), 3)
        for s in world.spawns
    }
    results.append(
        CheckResult(
            "RP2 every spawn is at least 2.5 m from every patrol path",
            all(v >= SPAWN_PATH_M - 1e-9 for v in spawn_path.values()),
            spawn_path,
        )
    )
    spawn_clear: dict[str, float] = {}
    for s in world.spawns:
        _, i = tree.query([(s.x, s.y)], k=8)
        gap = np.maximum(np.abs(centres[i[0]] - (s.x, s.y)) - g.h / 2, 0.0)
        spawn_clear[s.id] = round(float(np.hypot(gap[:, 0], gap[:, 1]).min()), 3)
    results.append(
        CheckResult(
            "RP2 every spawn is at least 1.5 m from walls and obstacles (raster)",
            all(v >= SPAWN_CLEARANCE_M - 1e-6 for v in spawn_clear.values()),
            spawn_clear,
        )
    )

    # plan §7: the office is reached only through D01 into the west room; the east room through D02
    widths, separates = door_openings(a, DOORS)
    results.append(
        CheckResult(
            "RP2 both office doors are 1.2 m at their listed centres",
            all(abs(w - DOOR_M) <= g.h + 1e-9 for w in widths),
            widths,
        )
    )
    reach: dict[str, dict[str, bool]] = {}
    s1 = world.spawns[0]
    for d in DOORS:
        labels, _ = ndimage.label(door_cut(a, d))
        main = labels[g.cell(s1.x, s1.y)]
        reach[d.id] = {rid: bool(labels[g.cell(*p)] == main) for rid, _, _, p in OFFICE_ROOMS}
    office_ok = (
        all(separates)
        and reach["D01"] == {"west": False, "east": False}
        and reach["D02"] == {"west": True, "east": False}
    )
    results.append(
        CheckResult(
            "RP2 office reached only through D01 (west room); east room only through D02",
            office_ok,
            {"separates": separates, "rooms connected to S1 with the door cut": reach},
        )
    )
    rooms = {rid: rect_area(a, x0, x1, y0, y1) for rid, (x0, x1), (y0, y1), _ in OFFICE_ROOMS}
    results.append(
        CheckResult(
            "RP2 office rooms: west 4.8 × 7.9 m less the cabinet, east 4.9 × 7.9 m",
            abs(rooms["west"] - (37.92 - 0.8)) < 1e-6 and abs(rooms["east"] - 38.71) < 1e-6,
            rooms,
        )
    )

    # plan §7: no slot narrower than 1.0 m between touching or adjacent boxes. A gap counts as a slot only
    # when the rectangle between the two boxes holds reachable floor (gaps sealed inside the outer wall and
    # its dock recesses are solid mass, not slots).
    bounds = [(p.name, p.footprint_bounds()) for p in world.static_primitives()]
    slots = []
    for i, (n1, b1) in enumerate(bounds):
        for n2, b2 in bounds[i + 1 :]:
            gap, (x0, x1, y0, y1) = _box_gap(b1, b2)
            if 1e-6 < gap < SLOT_M - 1e-6 and min(x1 - x0, y1 - y0) >= g.h - 1e-9:
                if rect_area(a, x0, x1, y0, y1) > 0.0:
                    slots.append(f"{n1}/{n2} {gap:.3f} m")
    results.append(
        CheckResult(
            "RP2 no slot narrower than 1.0 m between boxes",
            not slots,
            len(slots),
            "; ".join(slots[:6]),
        )
    )
    where = g.centre(*a.narrowest_cell) if a.narrowest_cell else None
    at_door = where is not None and any(
        math.hypot(where[0] - d.x, where[1] - d.y) <= DOOR_M / 2 + 0.2 for d in DOORS
    )
    results.append(
        CheckResult(
            "RP2 the narrowest passage lies in an office door",
            at_door,
            tuple(round(v, 2) for v in where) if where else None,
        )
    )

    # plan §3.3: aisle clear widths
    measured = {name: free_length(a, axis, at, lo, hi) for name, axis, at, lo, hi, _ in AISLES}
    results.append(
        CheckResult(
            "RP2 aisle widths match plan §3.3",
            all(abs(measured[name] - want) <= g.h + 1e-9 for name, *_, want in AISLES),
            measured,
        )
    )

    # plan §7: no third-party file or include; every primitive is a box we define
    kinds = sorted({p.kind.value for p in world.primitives})
    results.append(
        CheckResult(
            "RP2 no third-party file or include (primitives only, no sources)",
            kinds == [Kind.BOX.value] and not world.sources,
            {"kinds": kinds, "sources": len(world.sources)},
        )
    )
    return results


def build() -> World:
    xmin, xmax, ymin, ymax = HALL
    walls, notches = walled_rectangle(
        xmin,
        xmax,
        ymin,
        ymax,
        recesses={"s": [Recess(rid, x, DOCK_W, DOCK_D) for rid, x in DOCKS]},
        thickness=OUTER_T,
        height=OUTER_H,
    )
    racks = [
        box(n, x0, x1, y0, y1, RACK_H, Role.OBSTACLE, "furniture") for n, x0, x1, y0, y1 in RACKS
    ]
    columns = [
        box(
            n,
            x - COLUMN_HALF,
            x + COLUMN_HALF,
            y - COLUMN_HALF,
            y + COLUMN_HALF,
            COLUMN_H,
            Role.OBSTACLE,
            "core",
        )
        for n, x, y in COLUMNS
    ]
    office = [
        box(n, x0, x1, y0, y1, OFFICE_WALL_H, Role.WALL, "inner")
        for n, x0, x1, y0, y1 in OFFICE_WALLS
    ]
    loads = [box(n, x0, x1, y0, y1, h, role, mat) for n, x0, x1, y0, y1, h, role, mat in LOADS]
    return World(
        id="RP2",
        slug="industrial_warehouse",
        title="Industrial warehouse",
        tier="RP",
        role="evaluation",
        plan=PLAN,
        primitives=(*walls, *racks, *columns, *office, *loads),
        spawns=(
            Spawn("S1", -26.5, -13.0, 0.0, "west end of the staging area, facing the aisle ends"),
            Spawn("S2", 15.5, 11.0, math.pi, "east end of the north aisle, outside the office"),
            Spawn("S3", 25.5, -11.0, math.pi / 2, "east zone, south of the charger row"),
        ),
        expected=Expected(free_area_m2=1673.3, narrowest_passage_m=1.2, loops=10),
        notches=tuple(notches),
        doors=DOORS,
        dimensions=(
            Dimension((xmin, ymin - 1.4), (xmax, ymin - 1.4), "56.0 m"),
            Dimension((xmin - 1.0, ymin), (xmin - 1.0, ymax), "36.0 m"),
            Dimension((-19.0, 4.6), (-19.0, 7.1), "A3 2.5 m"),
            Dimension((-9.0, -4.0), (-5.0, -4.0), "4.0 m"),
        ),
        plan_data={
            "patrol_paths": {
                pid: {
                    "what": what,
                    "polylines_m": [[[x, y] for x, y in line] for line in lines],
                }
                for pid, what, lines in PATROL_PATHS
            },
            "patrol_clearance_m": PATH_CLEARANCE_M,
        },
        extra_checks=(_checks,),
    )
