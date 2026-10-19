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

"""P1 Pilot racetrack office. Pilot only; the base
world of PM1."""

from __future__ import annotations

import math

from ..checks import Analysis, no_symmetry
from ..geometry import Gap, Recess, box, door, wall_x, walled_rectangle
from ..layout_checks import door_openings, free_length, rect_area
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

DOOR_M = 1.2  # designed-world door width (maintainer, 18 October 2026)
# room id, interior x range, interior y range, door centre (x, y)
ROOMS = (
    ("NW", (-15.0, -5.1), (5.1, 10.0), (-10.0, 5.0)),
    ("N", (-4.9, 4.9), (5.1, 10.0), (1.5, 5.0)),
    ("NE", (5.1, 15.0), (5.1, 10.0), (9.0, 5.0)),
    ("BW", (-12.4, -0.1), (-7.4, 2.3), (-6.0, -7.5)),
    ("BE", (0.1, 12.4), (-7.4, 2.3), (7.0, 2.4)),
)
DOORS = tuple(door(f"D{i + 1:02d}", x, y, DOOR_M, "x") for i, (*_, (x, y)) in enumerate(ROOMS))
# ring side, axis of the measuring line, position, range: one line per side, clear of recesses and doors
RING = (
    ("south", "y", -3.0, -11.0, -7.5),
    ("north", "y", -3.0, 2.4, 5.0),
    ("west", "x", -5.0, -16.0, -12.5),
    ("east", "x", 0.0, 12.5, 16.0),
)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    results = [no_symmetry("P1 door and recess layout has no symmetry", world)]
    sizes = {rid: rect_area(a, x0, x1, y0, y1) for rid, (x0, x1), (y0, y1), _ in ROOMS}
    want = {rid: round((x1 - x0) * (y1 - y0), 4) for rid, (x0, x1), (y0, y1), _ in ROOMS}
    results.append(
        CheckResult(
            "P1 room interiors match the room table",
            all(abs(sizes[k] - want[k]) < 1e-6 for k in sizes),
            sizes,
        )
    )
    widths, separates = door_openings(a, DOORS)
    results.append(
        CheckResult(
            "P1 every door is 1.2 m at its listed centre",
            all(abs(w - DOOR_M) <= a.grid.h + 1e-9 for w in widths),
            widths,
        )
    )
    results.append(
        CheckResult(
            "P1 cutting each door separates exactly its room",
            all(separates),
            separates.count(True),
        )
    )
    ring = {side: free_length(a, axis, at, lo, hi) for side, axis, at, lo, hi in RING}
    results.append(
        CheckResult(
            "P1 ring is 2.4 m wide on every side",
            all(abs(v - 2.4) <= a.grid.h + 1e-9 for v in ring.values()),
            ring,
        )
    )
    return results


def build() -> World:
    walls, notches = walled_rectangle(
        -15.0,
        15.0,
        -10.0,
        10.0,
        recesses={
            "s": [Recess("L1", -9.0, 1.0, 0.4), Recess("L2", 4.0, 1.0, 0.4)],
            "e": [Recess("L3", -4.5, 1.0, 0.4)],
            "w": [Recess("L4", -1.5, 1.0, 0.4)],
        },
    )
    north_gaps = [Gap(x, DOOR_M) for rid, _, _, (x, _) in ROOMS if rid in ("NW", "N", "NE")]
    inner = [
        *wall_x("room_wall_n", 4.9, 5.1, -15.0, 15.0, north_gaps, 2.0, Role.WALL, "inner"),
        box("partition_w", -5.1, -4.9, 5.1, 10.0, 2.0, Role.WALL, "inner"),
        box("partition_e", 4.9, 5.1, 5.1, 10.0, 2.0, Role.WALL, "inner"),
        *wall_x("block_s", -7.6, -7.4, -12.6, 12.6, [Gap(-6.0, DOOR_M)], 2.0, Role.WALL, "inner"),
        *wall_x("block_n", 2.3, 2.5, -12.6, 12.6, [Gap(7.0, DOOR_M)], 2.0, Role.WALL, "inner"),
        box("block_w", -12.6, -12.4, -7.4, 2.3, 2.0, Role.WALL, "inner"),
        box("block_e", 12.4, 12.6, -7.4, 2.3, 2.0, Role.WALL, "inner"),
        box("block_divider", -0.1, 0.1, -7.4, 2.3, 2.0, Role.WALL, "inner"),
    ]
    return World(
        id="P1",
        slug="racetrack_office",
        title="Pilot racetrack office",
        tier="P",
        role="pilot",
        primitives=(*walls, *inner),
        spawns=(
            Spawn(
                "S1", 0.0, -8.8, 0.0, "middle of the south corridor; the ring continues both ways"
            ),
            Spawn("S2", -13.8, -4.0, math.pi / 2, "west corridor, facing the north-west corner"),
            Spawn("S3", -6.25, -2.55, -math.pi / 2, "centre of BW, looking through its door"),
        ),
        expected=Expected(free_area_m2=578.94, narrowest_passage_m=1.2, loops=1, loops_unsized=1),
        notches=tuple(notches),
        doors=DOORS,
        dimensions=(
            Dimension((-15.0, -11.2), (15.0, -11.2), "30.0 m"),
            Dimension((-3.0, -10.0), (-3.0, -7.6), "ring 2.4 m"),
            Dimension((-12.6, -1.0), (12.6, -1.0), "block 25.2 m"),
        ),
        extra_checks=(_checks,),
    )
