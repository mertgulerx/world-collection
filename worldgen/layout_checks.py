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

"""Measurements shared by the world-specific checks of room-and-door layouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from . import raster
from .checks import Analysis
from .world import BoolArray, Door

HALF_CELL = raster.RESOLUTION / 2  # keeps a measuring range inside its end cells


def rect_area(a: Analysis, x0: float, x1: float, y0: float, y1: float) -> float:
    """Reachable area (m²) inside the rectangle [x0, x1] × [y0, y1]."""
    g = a.grid
    r0, c0 = g.cell(x0 + HALF_CELL, y0 + HALF_CELL)
    r1, c1 = g.cell(x1 - HALF_CELL, y1 - HALF_CELL)
    return round(float(np.count_nonzero(a.reachable[r0 : r1 + 1, c0 : c1 + 1])) * g.h * g.h, 4)


def free_length(a: Analysis, axis: str, at: float, lo: float, hi: float) -> float:
    """Reachable length (m) along a grid line: x from lo to hi at y = at (axis "x"), or y at x = at."""
    g = a.grid
    if axis == "x":
        r, c0 = g.cell(lo + HALF_CELL, at)
        _, c1 = g.cell(hi - HALF_CELL, at)
        line = a.reachable[r, c0 : c1 + 1]
    else:
        r0, c = g.cell(at, lo + HALF_CELL)
        r1, _ = g.cell(at, hi - HALF_CELL)
        line = a.reachable[r0 : r1 + 1, c]
    return round(float(np.count_nonzero(line)) * g.h, 3)


def door_cut(a: Analysis, d: Door) -> BoolArray:
    """A copy of the reachable mask with the door's opening removed along the wall's centre line."""
    g = a.grid
    half = d.width / 2
    cut = a.reachable.copy()
    if d.wall_axis == "x":
        r, c0 = g.cell(d.x - half + HALF_CELL, d.y)
        _, c1 = g.cell(d.x + half - HALF_CELL, d.y)
        cut[r, c0 : c1 + 1] = False
    else:
        r0, c = g.cell(d.x, d.y - half + HALF_CELL)
        r1, _ = g.cell(d.x, d.y + half - HALF_CELL)
        cut[r0 : r1 + 1, c] = False
    return cut


def door_openings(a: Analysis, doors: Sequence[Door]) -> tuple[list[float], list[bool]]:
    """Per door: the free width through the wall's centre line within 0.5 m of the door, and whether
    cutting the opening adds exactly one component (the door separates exactly one room)."""
    base = ndimage.label(a.reachable)[1]
    widths, separates = [], []
    for d in doors:
        half = d.width / 2
        if d.wall_axis == "x":
            widths.append(free_length(a, "x", d.y, d.x - half - 0.5, d.x + half + 0.5))
        else:
            widths.append(free_length(a, "y", d.x, d.y - half - 0.5, d.y + half + 0.5))
        separates.append(ndimage.label(door_cut(a, d))[1] - base == 1)
    return widths, separates
