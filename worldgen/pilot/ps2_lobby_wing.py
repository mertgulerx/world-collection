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

"""P2 Pilot lobby and wing. Pilot only."""

from __future__ import annotations

import math

from ..checks import Analysis, no_symmetry
from ..geometry import Gap, Recess, box, door, wall_x, walled_rectangle
from ..layout_checks import door_openings, free_length, rect_area
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

DOOR_M = 1.2  # designed-world door width (maintainer, 18 October 2026)
PILLARS = (("pillar_1", -12.0, 4.0), ("pillar_2", -8.0, -2.5), ("pillar_3", -13.0, -6.0))
# room id, interior x range, interior y range, door centre (x, y)
ROOMS = (
    ("W1", (6.2, 11.0), (1.4, 9.0), (7.4, 1.3)),
    ("W2", (11.2, 16.0), (1.4, 9.0), (14.6, 1.3)),
    ("W3", (6.2, 11.0), (-9.0, -1.4), (8.9, -1.3)),
    ("W4", (11.2, 16.0), (-9.0, -1.4), (12.8, -1.3)),
)
DOORS = tuple(door(f"D{i + 1:02d}", x, y, DOOR_M, "x") for i, (*_, (x, y)) in enumerate(ROOMS))
# corridor, axis of the measuring line, position, range, expected width
CORRIDORS = (
    ("side_north", "y", 0.0, 2.9, 5.1, 2.0),
    ("side_south", "y", 0.0, -5.1, -2.9, 2.0),
    ("cross", "x", 2.0, 3.9, 6.1, 2.0),
    ("wing", "y", 10.0, -1.3, 1.3, 2.4),
)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    results = [no_symmetry("P2 pillar, door and recess layout has no symmetry", world)]
    lobby = rect_area(a, -16.0, -4.0, -9.0, 9.0)
    results.append(
        CheckResult(
            "P2 lobby is 12.0 × 18.0 m less three pillars",
            abs(lobby - (216.0 - 3 * 0.36)) < 1e-6,
            lobby,
        )
    )
    rooms = {rid: rect_area(a, x0, x1, y0, y1) for rid, (x0, x1), (y0, y1), _ in ROOMS}
    results.append(
        CheckResult(
            "P2 every room is 4.8 × 7.6 m",
            all(abs(v - 36.48) < 1e-6 for v in rooms.values()),
            rooms,
        )
    )
    widths, separates = door_openings(a, DOORS)
    results.append(
        CheckResult(
            "P2 every door is 1.2 m at its listed centre",
            all(abs(w - DOOR_M) <= a.grid.h + 1e-9 for w in widths),
            widths,
        )
    )
    results.append(
        CheckResult(
            "P2 cutting each door separates exactly its room",
            all(separates),
            separates.count(True),
        )
    )
    measured = {name: free_length(a, axis, at, lo, hi) for name, axis, at, lo, hi, _ in CORRIDORS}
    results.append(
        CheckResult(
            "P2 side and cross corridors 2.0 m, wing corridor 2.4 m",
            all(abs(measured[name] - want) <= a.grid.h + 1e-9 for name, *_, want in CORRIDORS),
            measured,
        )
    )
    return results


def build() -> World:
    walls, notches = walled_rectangle(
        -16.0,
        16.0,
        -9.0,
        9.0,
        recesses={"n": [Recess("L1", -9.5, 1.0, 0.4)], "w": [Recess("L2", -3.0, 1.0, 0.4)]},
    )
    north_gaps = [Gap(x, DOOR_M) for _, _, (y0, _), (x, _) in ROOMS if y0 > 0]
    south_gaps = [Gap(x, DOOR_M) for _, _, (y0, _), (x, _) in ROOMS if y0 < 0]
    inner = [
        box("core", -4.0, 4.0, -3.0, 3.0, 2.0, Role.WALL, "inner"),
        box("fill_n", -4.0, 6.0, 5.0, 9.0, 2.0, Role.WALL, "inner"),
        box("fill_s", -4.0, 6.0, -9.0, -5.0, 2.0, Role.WALL, "inner"),
        *wall_x("wing_wall_n", 1.2, 1.4, 6.0, 16.0, north_gaps, 2.0, Role.WALL, "inner"),
        *wall_x("wing_wall_s", -1.4, -1.2, 6.0, 16.0, south_gaps, 2.0, Role.WALL, "inner"),
        box("room_wall_nw", 6.0, 6.2, 1.4, 9.0, 2.0, Role.WALL, "inner"),
        box("room_wall_sw", 6.0, 6.2, -9.0, -1.4, 2.0, Role.WALL, "inner"),
        box("divider_n", 11.0, 11.2, 1.4, 9.0, 2.0, Role.WALL, "inner"),
        box("divider_s", 11.0, 11.2, -9.0, -1.4, 2.0, Role.WALL, "inner"),
    ]
    pillars = [
        box(n, x - 0.3, x + 0.3, y - 0.3, y + 0.3, 2.0, Role.OBSTACLE, "obstacle")
        for n, x, y in PILLARS
    ]
    return World(
        id="P2",
        slug="lobby_wing",
        title="Pilot lobby and wing",
        tier="P",
        role="pilot",
        primitives=(*walls, *inner, *pillars),
        spawns=(
            Spawn("S1", -12.0, 0.0, 0.0, "west lobby, facing the core 8.0 m ahead"),
            Spawn("S2", 13.6, 5.2, -math.pi / 2, "centre of W2, facing its corridor wall"),
            Spawn("S3", 11.0, 0.0, math.pi, "wing corridor, facing the cross corridor"),
        ),
        expected=Expected(free_area_m2=438.60, narrowest_passage_m=1.2, loops=1, loops_unsized=4),
        notches=tuple(notches),
        doors=DOORS,
        dimensions=(
            Dimension((-16.0, -10.2), (16.0, -10.2), "32.0 m"),
            Dimension((-15.0, -9.0), (-15.0, 9.0), "lobby 18.0 m"),
            Dimension((10.0, -1.2), (10.0, 1.2), "2.4 m"),
        ),
        extra_checks=(_checks,),
    )
