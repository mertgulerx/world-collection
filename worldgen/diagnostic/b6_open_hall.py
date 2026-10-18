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

"""SC6 Open hall (plan docs/plans/worlds/B6-open-hall.md)."""

from __future__ import annotations

import math

from ..checks import Analysis, no_symmetry
from ..geometry import box, walled_rectangle
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

# A 6 × 6 grid at 6 m spacing with offsets up to 1.0 m, drawn once and fixed here
# (maintainer, 18 October 2026: 36 pillars instead of 9; report §9 decision 49).
PILLARS = (
    ("p1", -15.0, -14.5),
    ("p2", -8.0, -14.5),
    ("p3", -3.0, -16.0),
    ("p4", 2.0, -15.5),
    ("p5", 9.5, -14.5),
    ("p6", 14.5, -15.5),
    ("p7", -15.5, -8.5),
    ("p8", -9.0, -8.5),
    ("p9", -3.0, -8.0),
    ("p10", 3.0, -10.0),
    ("p11", 8.0, -10.0),
    ("p12", 15.5, -8.0),
    ("p13", -16.0, -2.0),
    ("p14", -10.0, -4.0),
    ("p15", -3.0, -2.5),
    ("p16", 4.0, -2.0),
    ("p17", 9.5, -3.5),
    ("p18", 14.0, -2.5),
    ("p19", -15.5, 4.0),
    ("p20", -9.5, 2.5),
    ("p21", -4.0, 3.0),
    ("p22", 3.5, 4.0),
    ("p23", 9.5, 2.0),
    ("p24", 15.0, 3.5),
    ("p25", -15.0, 9.5),
    ("p26", -8.0, 8.0),
    ("p27", -2.0, 8.0),
    ("p28", 4.0, 9.0),
    ("p29", 9.0, 8.0),
    ("p30", 15.5, 10.0),
    ("p31", -15.0, 15.0),
    ("p32", -8.5, 15.5),
    ("p33", -2.5, 14.0),
    ("p34", 4.0, 14.0),
    ("p35", 9.5, 15.0),
    ("p36", 14.5, 16.0),
)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    largest = round(float(a.clearance[a.reachable].max()), 3)
    return [
        no_symmetry("SC6 pillar layout has no symmetry", world),
        CheckResult(
            "SC6 largest clearance 4.3 m ± 0.1 m", abs(largest - 4.3) <= 0.1 + 1e-9, largest
        ),
    ]


def build() -> World:
    walls, _ = walled_rectangle(-20.0, 20.0, -20.0, 20.0)
    pillars = [
        box(n, x - 0.3, x + 0.3, y - 0.3, y + 0.3, 2.0, Role.OBSTACLE, "obstacle")
        for n, x, y in PILLARS
    ]
    return World(
        id="SC6",
        slug="open_hall",
        title="Open hall",
        tier="SC",
        role="evaluation",
        plan="docs/plans/worlds/B6-open-hall.md",
        primitives=(*walls, *pillars),
        spawns=(
            Spawn("S1", 0.0, 0.0, 0.0, "hall centre; walls out of lidar range"),
            Spawn("S2", -17.0, -17.0, math.pi / 4, "south-west corner region, facing diagonally"),
            Spawn("S3", 17.0, -6.0, math.pi, "east side, facing into the hall"),
        ),
        expected=Expected(free_area_m2=1587.04, narrowest_passage_m=3.7, loops=0, loops_unsized=36),
        dimensions=(
            Dimension((-20.0, -21.0), (20.0, -21.0), "40.0 m"),
            Dimension((-3.0, -20.0), (-3.0, -16.3), "3.7 m"),
        ),
        sources=({"what": "layout idea (open map)", "cite": "faigl2015taskallocation"},),
        extra_checks=(_checks,),
    )
