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

"""F3 Warehouse aisles, a generated family of eight rack halls (plan docs/plans/worlds/C4-warehouse-aisles.md).

Lengths of the construction are integers in units of 0.1 m measured from the hall's south-west corner (the inner
faces of the walls), so every rack lies on the 0.1 m construction grid (plan §3.2).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..checks import Analysis, analyse
from ..geometry import Recess, grid_walls, walled_rectangle
from ..layout_checks import rect_area
from ..world import (
    BoolArray,
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    FamilyInfo,
    Notch,
    Role,
    Spawn,
    World,
)
from .common import (
    base_reject,
    first_failure,
    seed_rules,
    seeded_spawn,
    select,
    snap_units,
    spaced_centres,
    streams,
    walls_check,
)

FAMILY = "F3"
PLAN = "docs/plans/worlds/C4-warehouse-aisles.md"
SEED_BASE = 400000  # setting j tries seeds SEED_BASE + 1000·j + i (plan §3.3)
UNIT = 0.1
DEPTH = 20  # rack row depth D, 2.0 m
END_AISLE = 30  # 3.0 m between row ends and the end walls
CROSS = 30  # cross-aisle width, 3.0 m
RACK_RECESS_W = 10  # 1.0 m
RACK_RECESS_D = 3  # 0.3 m
WALL_RECESS_W = 1.0
WALL_RECESS_D = 0.4
WALL_RECESS_GAP = 30  # end-wall recess centres at least 3.0 m apart
WALL_RECESS_DRAWS = 1000  # one budget shared by the west and east walls
# setting j: hall W × H (m), aisle width a (units), cross-aisles c
SETTINGS = {
    1: (30, 20, 15, 1),
    2: (30, 20, 15, 3),
    3: (30, 20, 30, 1),
    4: (30, 20, 30, 3),
    5: (60, 40, 15, 1),
    6: (60, 40, 15, 3),
    7: (60, 40, 30, 1),
    8: (60, 40, 30, 3),
}
# planning ranges of plan §4.2: free area (m²); rack-face recesses are Binomial(2·n_r·(c + 1), 0.5)
RANGES = {
    1: (394.2, 396.6),
    2: (456.6, 460.2),
    3: (477.3, 479.1),
    4: (514.5, 517.5),
    5: (1288.1, 1292.3),
    6: (1426.1, 1431.5),
    7: (1694.3, 1697.3),
    8: (1781.9, 1786.1),
}
# Settings not built: the maintainer dropped settings 5, 6 and 8 (18 October 2026; report §9 decision 24).
DROPPED = frozenset({5, 6, 8})
BUILT = tuple(j for j in SETTINGS if j not in DROPPED)


def world_id(j: int) -> str:
    """The instance ID of setting j: its position among the built settings, so IDs have no gaps (report §9
    decision 51). Seeds still follow the setting number j."""
    return f"{FAMILY}-{BUILT.index(j) + 1:02d}"


FAMILY_INFO = FamilyInfo(
    plan=PLAN,
    instances=tuple(world_id(j) for j in BUILT),
    columns=4,
    seeds=seed_rules(SEED_BASE),
)


@dataclass(frozen=True, slots=True)
class Layout:
    width: int
    height: int
    aisle: int
    rows: tuple[tuple[int, int], ...]  # y0, y1 of each row, south to north
    segments: tuple[tuple[int, int], ...]  # x0, x1 shared by every row, west to east
    min_segment: int
    rack_recesses: tuple[tuple[int, int, str, int], ...]  # row, segment, face "s"/"n", centre x
    wall_recesses: Mapping[str, tuple[int, ...]]  # "w"/"e": centres y in draw order (read-only)


def _rows(height: int, aisle: int) -> list[tuple[int, int]]:
    """Plan §3.4 step 1: n_r rows; the leftover e widens the two wall aisles equally."""
    n_r = (height - aisle) // (DEPTH + aisle)
    leftover = height - (n_r + 1) * aisle - n_r * DEPTH
    start = aisle + leftover // 2
    return [(start + k * (DEPTH + aisle), start + k * (DEPTH + aisle) + DEPTH) for k in range(n_r)]


def _segments(
    width: int, cross: int, geom: np.random.Generator
) -> tuple[list[tuple[int, int]], int]:
    """Plan §3.4 steps 2–3: c + 1 segments of at least m_s, summing to L_r, separated by 3.0 m cross-aisles."""
    total = width - 2 * END_AISLE - CROSS * cross
    m_s = max(30, snap_units(0.5 * total * UNIT / (cross + 1)))
    weights = geom.dirichlet(np.ones(cross + 1))
    spare = total - (cross + 1) * m_s
    lengths = [m_s + math.floor(spare * float(w)) for w in weights[:-1]]
    lengths.append(total - sum(lengths))
    out = []
    x = END_AISLE
    for length in lengths:
        out.append((x, x + length))
        x += length + CROSS
    return out, m_s


def _layout(j: int, geom: np.random.Generator, land: np.random.Generator) -> Layout:
    w_m, h_m, aisle, cross = SETTINGS[j]
    width, height = 10 * w_m, 10 * h_m
    rows = _rows(height, aisle)
    segments, m_s = _segments(width, cross, geom)
    recesses = []
    for k in range(len(rows)):
        for s, (x0, x1) in enumerate(segments):
            for face in ("s", "n"):
                if land.random() < 0.5:
                    centre = snap_units(float(land.uniform((x0 + 10) * UNIT, (x1 - 10) * UNIT)))
                    recesses.append((k, s, face, centre))
    per_wall = round(h_m / 6)
    walls: dict[str, tuple[int, ...]] = {}
    budget = WALL_RECESS_DRAWS
    for side in ("w", "e"):
        centres, used = spaced_centres(
            land, per_wall, -h_m / 2 + 2.0, h_m / 2 - 2.0, WALL_RECESS_GAP, budget
        )
        budget -= used
        walls[side] = tuple(centres)
    return Layout(
        width,
        height,
        aisle,
        tuple(rows),
        tuple(segments),
        m_s,
        tuple(recesses),
        MappingProxyType(walls),
    )


def _occupancy(lay: Layout) -> BoolArray:
    occ = np.zeros((lay.height, lay.width), dtype=bool)
    for y0, y1 in lay.rows:
        for x0, x1 in lay.segments:
            occ[y0:y1, x0:x1] = True
    for k, s, face, centre in lay.rack_recesses:
        y0, y1 = lay.rows[k]
        a, b = centre - RACK_RECESS_W // 2, centre + RACK_RECESS_W // 2
        if face == "s":
            occ[y0 : y0 + RACK_RECESS_D, a:b] = False
        else:
            occ[y1 - RACK_RECESS_D : y1, a:b] = False
    return occ


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    layout: Layout


def _candidate(j: int, seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    w_m, h_m, aisle, cross = SETTINGS[j]
    geom, land, spawn = streams(seed)
    lay = _layout(j, geom, land)
    hx, hy = w_m / 2, h_m / 2
    outer, wall_notches = walled_rectangle(
        -hx,
        hx,
        -hy,
        hy,
        recesses={
            side: [
                Recess(f"{side.upper()}{i + 1}", c * UNIT, WALL_RECESS_W, WALL_RECESS_D)
                for i, c in enumerate(sorted(centres))
            ]
            for side, centres in lay.wall_recesses.items()
        },
    )
    racks = grid_walls(
        _occupancy(lay),
        UNIT,
        origin=(-hx, -hy),
        prefix="rack",
        role=Role.OBSTACLE,
        material="obstacle",
    )
    rack_notches = []
    for i, (k, _, face, centre) in enumerate(lay.rack_recesses):
        y0, y1 = lay.rows[k]
        ya, yb = (y0, y0 + RACK_RECESS_D) if face == "s" else (y1 - RACK_RECESS_D, y1)
        xa, xb = centre - RACK_RECESS_W // 2, centre + RACK_RECESS_W // 2
        poly = tuple(
            (round(x * UNIT - hx, 6), round(y * UNIT - hy, 6))
            for x, y in ((xa, ya), (xb, ya), (xb, yb), (xa, yb))
        )
        rack_notches.append(Notch(f"R{i + 1:03d}", poly))
    prims = (*outer, *racks)
    sx, sy, yaw = seeded_spawn(prims, (-hx + 1.5, 0.0), spawn)
    rack_area = sum(p.size[0] * p.size[1] for p in racks)
    notch_area = WALL_RECESS_W * WALL_RECESS_D * sum(len(c) for c in lay.wall_recesses.values())
    area = round(w_m * h_m - rack_area + notch_area, 4)
    loops = len(lay.rows) * (cross + 1)
    world = World(
        id=world_id(j),
        slug=f"aisles_h{w_m}x{h_m}_a{aisle * 10:03d}_x{cross}",
        title=f"Warehouse aisles {w_m}×{h_m} m, {aisle * UNIT:.1f} m aisles, {cross} cross-aisle"
        + ("s" if cross > 1 else ""),
        tier="F",
        role="evaluation",
        plan=PLAN,
        primitives=prims,
        spawns=(Spawn("S1", sx, sy, yaw, "seeded free cell, along the longest open axis ray"),),
        expected=Expected(
            free_area_m2=round(area, 2),
            narrowest_passage_m=aisle * UNIT,
            loops=loops,
            loops_unsized=loops,
            minimum_passage_m=aisle * UNIT,
        ),
        family=FAMILY,
        parameters={
            "hall_m": [w_m, h_m],
            "aisle_width_m": aisle * UNIT,
            "cross_aisles": cross,
            "rack_depth_m": DEPTH * UNIT,
            "rows": len(lay.rows),
        },
        seed=seed,
        notches=(*wall_notches, *rack_notches),
        dimensions=(
            Dimension((-hx, -hy - 0.9), (hx, -hy - 0.9), f"{w_m:.1f} m"),
            Dimension((-hx - 0.9, -hy), (-hx - 0.9, hy), f"{h_m:.1f} m"),
        ),
        sources=(
            {"what": "loops and revisits", "cite": "xu2022explorebench"},
            {"what": "narrow passages", "cite": "wettach2016experimental"},
        ),
        extra_checks=(_checks(j, lay),),
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "rows_y_m": [
                [round(y0 * UNIT - hy, 2), round(y1 * UNIT - hy, 2)] for y0, y1 in lay.rows
            ],
            "segments_x_m": [
                [round(x0 * UNIT - hx, 2), round(x1 * UNIT - hx, 2)] for x0, x1 in lay.segments
            ],
            "min_segment_m": round(lay.min_segment * UNIT, 2),
            "rack_recesses": len(lay.rack_recesses),
            "wall_recesses": {
                s: [round(c * UNIT, 2) for c in sorted(v)] for s, v in lay.wall_recesses.items()
            },
            "construction_free_area_m2": area,
            "sheet_label": f"{len(lay.rows)} rows, loops {loops}, {len(lay.rack_recesses)} rack recesses",
        },
    )
    return Candidate(world, lay)


def _reject_reason(j: int, c: Candidate, a: Analysis) -> str | None:
    """Why candidate `c` of setting j is rejected during selection, or None to accept it.

    Selection runs the shared rules of `common.base_reject` (passages of at least the aisle width, loops
    n_r·(c + 1)) and the plan §7 family checks except the planning-range check, which only stops the build for
    review. The build runs everything else on the accepted candidate: `gz sdf -k`, the free-area tolerance,
    heights, determinism (T0 checks 1, 5, 6 and 8) and the planning-range check.
    """
    return base_reject(c.world, a) or first_failure(
        _family_checks(j, c.layout, c.world, a, ranges=False)
    )


@functools.cache
def instance(j: int) -> Candidate:
    """The first accepted candidate of setting j (seed SEED_BASE + 1000·j + i)."""
    return select(
        world_id(j),
        SEED_BASE + 1000 * j,
        functools.partial(_candidate, j),
        lambda c: _reject_reason(j, c, analyse(c.world)),
    )


def _family_checks(
    j: int, lay: Layout, world: World, a: Analysis, *, ranges: bool
) -> list[CheckResult]:
    """The family checks of plan §7 on one candidate; `ranges` adds the planning-range check."""
    _, _, aisle, cross = SETTINGS[j]
    hx, hy = lay.width / 2 * UNIT, lay.height / 2 * UNIT
    span = (END_AISLE * UNIT - hx, hx - END_AISLE * UNIT)
    results: list[CheckResult] = []
    bands = [(0, lay.rows[0][0])]
    bands += [(lay.rows[k][1], lay.rows[k + 1][0]) for k in range(len(lay.rows) - 1)]
    bands.append((lay.rows[-1][1], lay.height))
    widths = [round((b - a_) * UNIT, 2) for a_, b in bands]
    clear = [
        abs(
            rect_area(a, span[0], span[1], a_ * UNIT - hy, b * UNIT - hy)
            - (b - a_) * UNIT * (span[1] - span[0])
        )
        < 1e-6
        for a_, b in bands
    ]
    wall_aisle = (lay.height - len(lay.rows) * DEPTH - (len(lay.rows) - 1) * aisle) / 2 * UNIT
    results.append(
        CheckResult(
            "F3 rows: interior aisles exactly a, wall aisles a + e/2, every aisle clear",
            all(abs(w - aisle * UNIT) < 1e-9 for w in widths[1:-1])
            and all(abs(w - wall_aisle) < 1e-9 for w in (widths[0], widths[-1]))
            and all(clear),
            {"rows": len(lay.rows), "aisles_m": sorted(set(widths)), "clear": all(clear)},
        )
    )
    lengths = [x1 - x0 for x0, x1 in lay.segments]
    total = lay.width - 2 * END_AISLE - CROSS * cross
    gaps = [lay.segments[i + 1][0] - lay.segments[i][1] for i in range(len(lay.segments) - 1)]
    cross_clear = [
        abs(rect_area(a, x0 * UNIT - hx, x1 * UNIT - hx, -hy, hy) - (x1 - x0) * UNIT * 2 * hy)
        < 1e-6
        for x0, x1 in [(0, END_AISLE), (lay.width - END_AISLE, lay.width)]
        + [(lay.segments[i][1], lay.segments[i + 1][0]) for i in range(len(lay.segments) - 1)]
    ]
    results.append(
        CheckResult(
            "F3 segments: c + 1 per row, each ≥ m_s, summing to L_r; cross and end aisles 3.0 m and clear",
            len(lay.segments) == cross + 1
            and min(lengths) >= lay.min_segment
            and sum(lengths) == total
            and all(g == CROSS for g in gaps)
            and all(cross_clear),
            {
                "segments_m": [round(v * UNIT, 1) for v in lengths],
                "min_segment_m": round(lay.min_segment * UNIT, 1),
                "clear": all(cross_clear),
            },
        )
    )
    ends = [
        min(centre - lay.segments[s][0], lay.segments[s][1] - centre) * UNIT
        for _, s, _, centre in lay.rack_recesses
    ]
    wall_gaps = [
        abs(p - q) * UNIT
        for centres in lay.wall_recesses.values()
        for i, p in enumerate(centres)
        for q in centres[i + 1 :]
    ]
    results.append(
        CheckResult(
            "F3 recesses: rack recesses ≥ 1.0 m from segment ends, end-wall recesses ≥ 3.0 m apart",
            (not ends or min(ends) >= 1.0 - 1e-9) and min(wall_gaps) >= 3.0 - 1e-9,
            {
                "rack_recesses": len(lay.rack_recesses),
                "min_end_m": round(min(ends, default=math.inf), 2),
                "min_wall_gap_m": round(min(wall_gaps), 2),
            },
        )
    )
    racks = [p for p in world.primitives if p.role is Role.OBSTACLE]
    results.append(walls_check(FAMILY, world, (Role.OBSTACLE, Role.BOUNDARY_WALL)))
    results.append(
        CheckResult(
            "F3 racks 2.0 m high from z = 0",
            all(abs(p.height - 2.0) < 1e-9 and p.z_min == 0.0 for p in racks),
            len(racks),
        )
    )
    if ranges:
        lo, hi = RANGES[j]
        area = round(a.free_area_m2, 2)
        results.append(
            CheckResult(
                "F3 free area inside the plan's planning range (outside: stop for review)",
                lo <= area <= hi,
                area,
                f"range {RANGES[j]}",
            )
        )
    return results


def _checks(j: int, lay: Layout) -> ExtraCheck:
    """The build's family checks, bound to this candidate's layout."""

    def run(world: World, a: Analysis) -> list[CheckResult]:
        return _family_checks(j, lay, world, a, ranges=True)

    return run


def build(j: int) -> World:
    return instance(j).world


def builders() -> dict[str, Callable[[], World]]:
    return {world_id(j): functools.partial(build, j) for j in BUILT}
