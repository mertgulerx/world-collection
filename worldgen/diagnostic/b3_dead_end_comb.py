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

"""SC3 Dead-end comb."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..checks import Analysis, no_symmetry
from ..geometry import box, door, walled_rectangle
from ..layout_checks import HALF_CELL, door_cut, free_length
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World

# (id, side, x0, x1, end face y): branch length is measured from the spine axis y = 0
BRANCHES = (
    ("BN1", "n", -20.0, -18.0, 9.0),
    ("BN2", "n", -10.0, -8.0, 6.0),
    ("BN3", "n", 0.0, 2.0, 10.0),
    ("BN4", "n", 10.0, 12.0, 7.0),
    ("BS1", "s", -15.0, -13.0, -8.0),
    ("BS2", "s", -5.0, -3.0, -10.0),
    ("BS3", "s", 5.0, 7.0, -6.0),
    ("BS4", "s", 18.0, 20.0, -7.0),
)
FILLS = (
    ("fn1", -18.0, -10.0, 1.5, 10.0),
    ("fn2", -8.0, 0.0, 1.5, 10.0),
    ("fn3", 2.0, 10.0, 1.5, 10.0),
    ("fn4", 12.0, 20.0, 1.5, 10.0),
    ("fn5", -20.0, -18.0, 9.0, 10.0),
    ("fn6", -10.0, -8.0, 6.0, 10.0),
    ("fn7", 10.0, 12.0, 7.0, 10.0),
    ("fs1", -20.0, -15.0, -10.0, -1.5),
    ("fs2", -13.0, -5.0, -10.0, -1.5),
    ("fs3", -3.0, 5.0, -10.0, -1.5),
    ("fs4", 7.0, 18.0, -10.0, -1.5),
    ("fs5", -15.0, -13.0, -10.0, -8.0),
    ("fs6", 5.0, 7.0, -10.0, -6.0),
    ("fs7", 18.0, 20.0, -10.0, -7.0),
)


def _checks(world: World, a: Analysis) -> list[CheckResult]:
    g = a.grid
    results = [no_symmetry("SC3 branch layout has no symmetry", world)]
    base = ndimage.label(a.reachable)[1]
    separated, widths, ends = [], [], []
    for bid, side, x0, x1, end in BRANCHES:
        # the mouth is the first cell row of the branch beyond the spine face y = ±1.5
        mouth_y = 1.5 if side == "n" else -1.5 - HALF_CELL
        mouth = door(bid, (x0 + x1) / 2, mouth_y, x1 - x0, "x")
        separated.append(ndimage.label(door_cut(a, mouth))[1] - base == 1)
        # half-way along the branch, with 0.25 m of the fills on either side in the measuring range
        widths.append(free_length(a, "x", (1.5 + end) / 2, x0 - 0.25, x1 + 0.25))
        _, cx = g.cell((x0 + x1) / 2, 0.0)
        column = np.nonzero(a.reachable[:, cx])[0]
        far = (column.max() + 1) if side == "n" else column.min()
        ends.append(round(float(g.y0 + far * g.h), 3))
    results.append(
        CheckResult(
            "SC3 cutting each branch mouth separates exactly that branch",
            all(separated),
            separated.count(True),
        )
    )
    results.append(
        CheckResult(
            "SC3 every branch is 2.0 m wide",
            all(abs(w - 2.0) <= g.h + 1e-9 for w in widths),
            sorted(set(widths)),
        )
    )
    expected_ends = [b[4] for b in BRANCHES]
    ok = all(abs(e - x) <= g.h + 1e-9 for e, x in zip(ends, expected_ends, strict=True))
    results.append(CheckResult("SC3 branch end faces at the listed y", ok, ends))
    return results


def build() -> World:
    walls, _ = walled_rectangle(-20.0, 20.0, -10.0, 10.0)
    fills = [box(name, x0, x1, y0, y1, 2.0, Role.WALL, "inner") for name, x0, x1, y0, y1 in FILLS]
    return World(
        id="SC3",
        slug="dead_end_comb",
        title="Dead-end comb",
        tier="SC",
        role="evaluation",
        primitives=(*walls, *fills),
        spawns=(
            Spawn("S1", -17.0, 0.0, 0.0, "west end of the spine, facing along it"),
            Spawn("S2", 3.5, 0.0, math.pi, "middle of the spine, between BN3 and BS3"),
            Spawn("S3", 1.0, 8.0, -math.pi / 2, "inside BN3, 2.0 m from its end, facing out"),
        ),
        expected=Expected(free_area_m2=222.0, narrowest_passage_m=2.0, loops=0, loops_unsized=0),
        dimensions=(
            Dimension((-20.0, -10.8), (20.0, -10.8), "40.0 m"),
            Dimension((-11.5, -1.5), (-11.5, 1.5), "spine 3.0 m"),
            Dimension((10.0, 4.0), (12.0, 4.0), "2.0 m"),
        ),
        sources=(
            {"what": "layout idea (maze and corridor maps)", "cite": "yan2016benchmarking"},
            {"what": "layout idea (corridor and maze worlds)", "cite": "liu2025enhancing"},
        ),
        extra_checks=(_checks,),
        plan_data={"branches": {b[0]: {"x": [b[2], b[3]], "end_y": b[4]} for b in BRANCHES}},
    )
