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

"""SC4 Office strip (plan docs/plans/worlds/B4-office-strip.md). Also the base world of E1."""

from __future__ import annotations

import math

from ..checks import Analysis, no_symmetry
from ..geometry import Gap, box, door, walled_rectangle, wall_x
from ..layout_checks import HALF_CELL, door_openings, rect_area
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

# room id, side, room x range, door centre x
ROOMS = (
    ("R01", "s", (-12.9, -7.9), -11.8),
    ("R02", "s", (-7.7, -2.7), -4.2),
    ("R03", "s", (-2.5, 2.5), 0.0),
    ("R04", "s", (2.7, 7.7), 4.7),
    ("R05", "s", (7.9, 12.9), 11.8),
    ("R06", "n", (-12.9, -7.9), -9.9),
    ("R07", "n", (-7.7, -2.7), -6.6),
    ("R08", "n", (-2.5, 2.5), 1.4),
    ("R09", "n", (2.7, 7.7), 5.2),
    ("R10", "n", (7.9, 12.9), 9.4),
)
PARTITIONS = (-7.9, -2.7, 2.5, 7.7)  # west faces; each partition is 0.2 m thick
DOOR_M = 1.2  # designed-world door width (maintainer, 18 October 2026: Nav2 struggled with 1.0 m)
CORRIDOR_X = (-12.9, 12.9)  # inner faces of the outer end walls (the corridor's ends)
CORRIDOR_Y = (-1.5, 1.5)  # corridor inner faces
DOOR_WALL_Y = {"s": -1.6, "n": 1.6}  # centre lines of the corridor walls carrying the doors
DOORS = tuple(
    door(f"D{i + 1:02d}", cx, DOOR_WALL_Y[side], DOOR_M, "x")
    for i, (_, side, _, cx) in enumerate(ROOMS)
)


def room_y(side: str) -> tuple[float, float]:
    """Interior y range of the rooms on side "s" or "n"."""
    return (-6.7, -1.7) if side == "s" else (1.7, 6.7)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    g = a.grid
    results = [no_symmetry("SC4 door offsets have no symmetry", world)]
    sizes = [rect_area(a, x0, x1, *room_y(side)) for _, side, (x0, x1), _ in ROOMS]
    widths, separated = door_openings(a, DOORS)
    doors_ok = [abs(w - DOOR_M) < g.h / 2 for w in widths]
    results.append(
        CheckResult(
            "SC4 every room is 5.0 × 5.0 m",
            all(abs(s - 25.0) < 1e-6 for s in sizes),
            sorted(set(sizes)),
        )
    )
    results.append(
        CheckResult(
            "SC4 every door is 1.2 m at its listed position", all(doors_ok), doors_ok.count(True)
        )
    )
    results.append(
        CheckResult(
            "SC4 cutting each door separates exactly its room",
            all(separated),
            separated.count(True),
        )
    )
    r0, c0 = g.cell(CORRIDOR_X[0] + HALF_CELL, CORRIDOR_Y[0] + HALF_CELL)
    r1, c1 = g.cell(CORRIDOR_X[1] - HALF_CELL, CORRIDOR_Y[1] - HALF_CELL)
    clear = bool(a.reachable[r0 : r1 + 1, c0 : c1 + 1].all())
    results.append(CheckResult("SC4 corridor is clear", clear, clear))
    return results


def build() -> World:
    xmin, xmax = CORRIDOR_X
    walls, _ = walled_rectangle(xmin, xmax, room_y("s")[0], room_y("n")[1])
    inner = []
    for side, (y0, y1) in (
        ("s", (room_y("s")[1], CORRIDOR_Y[0])),
        ("n", (CORRIDOR_Y[1], room_y("n")[0])),
    ):
        gaps = [Gap(cx, DOOR_M) for _, s, _, cx in ROOMS if s == side]
        inner += wall_x(f"corridor_{side}", y0, y1, xmin, xmax, gaps, 2.0, Role.WALL, "inner")
        ry0, ry1 = room_y(side)
        for i, x in enumerate(PARTITIONS):
            inner.append(
                box(f"partition_{side}_{i}", x, x + 0.2, ry0, ry1, 2.0, Role.WALL, "inner")
            )
    return World(
        id="SC4",
        slug="office_strip",
        title="Office strip",
        tier="SC",
        role="evaluation",
        plan="docs/plans/worlds/B4-office-strip.md",
        primitives=(*walls, *inner),
        spawns=(
            Spawn("S1", -11.0, 0.0, 0.0, "west end of the corridor, every room ahead"),
            Spawn("S2", -1.0, 0.0, 0.0, "middle of the corridor"),
            Spawn("S3", 5.2, 4.2, -math.pi / 2, "centre of room R09, facing its door"),
        ),
        expected=Expected(free_area_m2=329.8, narrowest_passage_m=1.2, loops=0, loops_unsized=0),
        doors=DOORS,
        dimensions=(
            Dimension((-12.9, -7.6), (12.9, -7.6), "25.8 m"),
            Dimension((10.4 - 2.5, 4.2), (10.4 + 2.5, 4.2), "room 5.0 m"),
            Dimension((-6.0, -1.5), (-6.0, 1.5), "corridor 3.0 m"),
        ),
        sources=(
            {"what": "layout idea (rooms scene)", "cite": "xu2022explorebench"},
            {"what": "layout idea (many-room and office maps)", "cite": "holz2010evaluating"},
        ),
        extra_checks=(_checks,),
    )
