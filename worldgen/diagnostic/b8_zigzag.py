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

"""SC8 Zigzag."""

from __future__ import annotations

import math

from scipy import ndimage

from .. import raster
from ..checks import Analysis, no_symmetry
from ..geometry import box, door, walled_rectangle
from ..layout_checks import door_cut, free_length
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

# id, x range, y range (as PILASTERS)
DIVIDERS = (
    ("d1", -15.0, 11.0, -6.4, -6.2),
    ("d2", -11.0, 15.0, -2.2, -2.0),
    ("d3", -15.0, 11.0, 2.0, 2.2),
    ("d4", -11.0, 15.0, 6.2, 6.4),
)
PILASTERS = (
    ("k1", -4.3, -3.7, -10.4, -10.1),
    ("k2", 7.2, 7.8, -6.7, -6.4),
    ("k3", -8.8, -8.2, -6.2, -5.9),
    ("k4", 2.7, 3.3, 1.7, 2.0),
    ("k5", -2.8, -2.2, 2.2, 2.5),
    ("k6", 5.7, 6.3, 5.9, 6.2),
    ("k7", 5.2, 5.8, 10.1, 10.4),
)
# leg y ranges, south to north
LEGS = ((-10.4, -6.4), (-6.2, -2.2), (-2.0, 2.0), (2.2, 6.2), (6.4, 10.4))
LEG_X = -1.0  # leg widths are measured at x = −1, clear of every pilaster


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    g = a.grid
    results = [no_symmetry("SC8 pilaster layout has no symmetry", world)]
    s1, s2, s3 = world.spawns
    d1, d3 = raster.geodesic(
        a.reachable, g.cell(s2.x, s2.y), [g.cell(s1.x, s1.y), g.cell(s3.x, s3.y)]
    )
    results.append(
        CheckResult(
            "SC8 S2 is equally far from both ends (< 1.0 m apart)",
            abs(d1 - d3) < 1.0,
            {"S2-S1 m": round(d1, 2), "S2-S3 m": round(d3, 2)},
        )
    )
    widths = [free_length(a, "y", LEG_X, y0, y1) for y0, y1 in LEGS]
    results.append(
        CheckResult(
            "SC8 every leg is 4.0 m wide", all(abs(w - 4.0) <= g.h + 1e-9 for w in widths), widths
        )
    )
    l3_y0, l3_y1 = LEGS[2]
    cut = door_cut(a, door("L3", LEG_X, (l3_y0 + l3_y1) / 2, l3_y1 - l3_y0, "y"))
    parts = ndimage.label(cut)[1]
    results.append(CheckResult("SC8 cutting a leg splits the path in two", parts == 2, parts))
    return results


def build() -> World:
    walls, _ = walled_rectangle(-15.0, 15.0, -10.4, 10.4)
    dividers = [box(n, x0, x1, y0, y1, 2.0, Role.WALL, "inner") for n, x0, x1, y0, y1 in DIVIDERS]
    pilasters = [
        box(n, x0, x1, y0, y1, 2.0, Role.LANDMARK, "landmark") for n, x0, x1, y0, y1 in PILASTERS
    ]
    return World(
        id="SC8",
        slug="zigzag",
        title="Zigzag",
        tier="SC",
        role="evaluation",
        primitives=(*walls, *dividers, *pilasters),
        spawns=(
            Spawn("S1", -13.5, -8.4, 0.0, "west end of L1, facing along the path"),
            Spawn("S2", 0.0, 0.0, 0.0, "middle of L3, the midpoint of the path"),
            Spawn("S3", 13.5, 8.4, math.pi, "east end of L5, facing along the path"),
        ),
        expected=Expected(free_area_m2=601.94, narrowest_passage_m=3.7, loops=0, loops_unsized=0),
        dimensions=(
            Dimension((-15.0, -11.2), (15.0, -11.2), "30.0 m"),
            Dimension((-12.0, -10.4), (-12.0, -6.4), "4.0 m"),
        ),
        sources=({"what": "layout idea (zigzag map)", "cite": "yan2016benchmarking"},),
        extra_checks=(_checks,),
    )
