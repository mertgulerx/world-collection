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

"""SC2 Narrow passage, variants B2a (1.2 m corridor) and B2b (2.0 m)."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..checks import Analysis, no_symmetry
from ..geometry import Recess, straight_wall, wall_x
from ..world import (
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    Notch,
    Primitive,
    Role,
    Spawn,
    World,
)

T = 0.2
VARIANTS = {"SC2a": (1.2, "narrow_passage_1m2", 459.2), "SC2b": (2.0, "narrow_passage_2m0", 464.0)}


def _walls(w: float) -> tuple[list[Primitive], list[Notch]]:
    half = w / 2
    prims: list[Primitive] = []
    notches: list[Notch] = []
    for args in (
        ("w_hall", "s", -7.5, -18.0 - T, -3.0 + T, [Recess("L1", -13.0, 1.0, 0.4)]),
        ("w_hall", "n", 7.5, -18.0 - T, -3.0 + T, []),
        ("w_hall", "w", -18.0, -7.5, 7.5, [Recess("L2", 3.5, 1.0, 0.4)]),
        ("w_hall_upper", "e", -3.0, half, 7.5, []),
        ("w_hall_lower", "e", -3.0, -7.5, -half, []),
        ("e_hall", "s", -7.5, 3.0 - T, 18.0 + T, [Recess("L5", 14.0, 1.0, 0.4)]),
        ("e_hall", "n", 7.5, 3.0 - T, 18.0 + T, [Recess("L3", 8.0, 1.0, 0.4)]),
        ("e_hall", "e", 18.0, -7.5, 7.5, [Recess("L4", -4.5, 1.0, 0.4)]),
        ("e_hall_upper", "w", 3.0, half, 7.5, []),
        ("e_hall_lower", "w", 3.0, -7.5, -half, []),
    ):
        prefix, side, inner, a0, a1, recesses = args
        walls, found = straight_wall(prefix, side, inner, a0, a1, recesses)
        prims += walls
        notches += found
    prims += wall_x(
        "corridor_n",
        half,
        half + T,
        -3.0 + T,
        3.0 - T,
        role=Role.BOUNDARY_WALL,
        material="boundary",
    )
    prims += wall_x(
        "corridor_s",
        -half - T,
        -half,
        -3.0 + T,
        3.0 - T,
        role=Role.BOUNDARY_WALL,
        material="boundary",
    )
    return prims, sorted(notches, key=lambda n: n.id)


def _checks(width: float) -> ExtraCheck:
    def run(world: World, a: Analysis) -> list[CheckResult]:
        g = a.grid
        results = [no_symmetry("SC2 recess layout has no symmetry", world)]
        _, c_lo = g.cell(-3.0, 0.0)
        _, c_hi = g.cell(3.0, 0.0)
        cut = a.reachable.copy()
        cut[:, c_lo : c_hi + 1] = False
        _, parts = ndimage.label(cut)
        results.append(
            CheckResult("SC2 removing the corridor leaves exactly two halls", parts == 2, parts)
        )
        corridor = [
            round(float(np.count_nonzero(a.reachable[:, g.cell(float(x), 0.0)[1]])) * g.h, 3)
            for x in np.arange(-2.5, 2.51, 0.5)
        ]
        ok = all(abs(v - width) <= g.h + 1e-9 for v in corridor)
        results.append(
            CheckResult(
                f"SC2 corridor is {width} m wide across x −2.5…2.5", ok, sorted(set(corridor))
            )
        )
        where = g.centre(*a.narrowest_cell) if a.narrowest_cell else (math.inf, math.inf)
        results.append(
            CheckResult(
                "SC2 narrowest passage lies in the corridor",
                abs(where[0]) <= 3.0,
                tuple(round(v, 2) for v in where),
            )
        )
        return results

    return run


def build(variant: str) -> World:
    width, slug, area = VARIANTS[variant]
    prims, notches = _walls(width)
    return World(
        id=variant,
        slug=slug,
        title=f"Narrow passage, {width:.1f} m corridor",
        tier="SC",
        role="evaluation",
        primitives=tuple(prims),
        spawns=(
            Spawn("S1", -10.5, 0.0, 0.0, "west hall centre, facing the corridor mouth"),
            Spawn("S2", -15.5, -5.0, math.pi / 2, "west hall, south-west; corridor out of range"),
            Spawn("S3", 11.0, 4.0, math.pi, "east hall, off-axis view of the mouth"),
        ),
        expected=Expected(free_area_m2=area, narrowest_passage_m=width, loops=0, loops_unsized=0),
        parameters={"corridor_width_m": width},
        notches=tuple(notches),
        dimensions=(
            Dimension((-18.0, -8.6), (-3.0, -8.6), "15.0 m"),
            Dimension((3.0, -8.6), (18.0, -8.6), "15.0 m"),
            Dimension((-3.0, 2.2), (3.0, 2.2), "corridor 6.0 m"),
        ),
        sources=({"what": "layout idea (narrow-corridor scene)", "cite": "xu2022explorebench"},),
        extra_checks=(_checks(width),),
    )


def build_a() -> World:
    return build("SC2a")


def build_b() -> World:
    return build("SC2b")
