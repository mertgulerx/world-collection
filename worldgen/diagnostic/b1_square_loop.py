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

"""SC1 Square loop."""

from __future__ import annotations

import math

from ..checks import Analysis, no_symmetry
from ..geometry import Recess, box, walled_rectangle
from ..world import CheckResult, Dimension, Expected, Role, Spawn, World


def _checks(world: World, analysis: Analysis) -> list[CheckResult]:
    return [no_symmetry("SC1 recess layout has no symmetry", world)]


def build() -> World:
    walls, notches = walled_rectangle(
        -15.0,
        15.0,
        -15.0,
        15.0,
        recesses={
            "s": [Recess("L1", -6.0, 1.0, 0.4), Recess("L2", 4.5, 1.0, 0.4)],
            "e": [Recess("L3", -2.0, 1.0, 0.4)],
            "n": [Recess("L4", 8.0, 1.0, 0.4)],
            "w": [Recess("L5", 5.5, 1.0, 0.4)],
        },
    )
    core = box("core", -12.0, 12.0, -12.0, 12.0, 2.0, Role.WALL, "core")
    return World(
        id="SC1",
        slug="square_loop",
        title="Square loop",
        tier="SC",
        role="evaluation",
        primitives=(*walls, core),
        spawns=(
            Spawn("S1", 0.0, -13.5, 0.0, "middle of the south corridor"),
            Spawn("S2", -13.5, 0.0, math.pi / 2, "middle of the west corridor"),
            Spawn("S3", 13.5, 13.5, math.pi, "north-east corner"),
        ),
        expected=Expected(free_area_m2=326.0, narrowest_passage_m=3.0, loops=1, loops_unsized=1),
        notches=tuple(notches),
        dimensions=(
            Dimension((-15.0, -16.6), (15.0, -16.6), "30.0 m"),
            Dimension((-12.0, 13.0), (12.0, 13.0), "core 24.0 m"),
            Dimension((12.0, 6.0), (15.0, 6.0), "3.0 m"),
        ),
        sources=(
            {"what": "layout idea (loop scene)", "cite": "xu2022explorebench"},
            {"what": "layout idea (loop map)", "cite": "yan2016benchmarking"},
        ),
        extra_checks=(_checks,),
    )
