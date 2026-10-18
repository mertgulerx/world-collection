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

"""Primitive builders (T0 §5). Every builder states the exact geometry it returns."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .world import BoolArray, Door, Kind, Notch, Point, Primitive, Role

EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Recess:
    """A notch in a wall: `centre` along the wall, `width` along it, `depth` away from the free side."""

    id: str
    centre: float
    width: float
    depth: float


@dataclass(frozen=True, slots=True)
class Gap:
    """A door gap in a wall segment: `centre` along the wall and clear `width`."""

    centre: float
    width: float


def box(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    height: float,
    role: Role,
    material: str,
) -> Primitive:
    """An axis-aligned box covering [x0, x1] × [y0, y1] × [0, height]."""
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        raise ValueError(f"box {name} is empty: x {x0}..{x1}, y {y0}..{y1}")
    return Primitive(
        Kind.BOX,
        name,
        round((x0 + x1) / 2, 6),
        round((y0 + y1) / 2, 6),
        0.0,
        (round(x1 - x0, 6), round(y1 - y0, 6), height),
        role,
        material,
    )


def cylinder(
    name: str, x: float, y: float, radius: float, height: float, role: Role, material: str
) -> Primitive:
    """A vertical cylinder of `radius` centred at (x, y), from z = 0 to `height`."""
    return Primitive(Kind.CYLINDER, name, x, y, 0.0, (radius, height), role, material)


def prism(
    name: str, vertices: Sequence[Point], height: float, role: Role, material: str
) -> Primitive:
    """A vertical prism over a simple counter-clockwise polygon (world frame, vertices rounded to 1 mm), from
    z = 0 to `height`. Its reference point (the SDF pose) is the centre of the polygon's bounding box, rounded
    to 1 mm; the World validation rejects polygons that are not simple, not counter-clockwise or have edges
    shorter than 2 cm."""
    pts = tuple((round(float(x), 3), round(float(y), 3)) for x, y in vertices)
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    cx = round((min(xs) + max(xs)) / 2, 3)
    cy = round((min(ys) + max(ys)) / 2, 3)
    return Primitive(Kind.POLYGON, name, cx, cy, 0.0, (height,), role, material, vertices=pts)


def _segments(start: float, end: float, gaps: Sequence[Gap]) -> list[tuple[float, float]]:
    """The parts of [start, end] left after removing every gap, in order."""
    parts: list[tuple[float, float]] = []
    cursor = start
    for gap in sorted(gaps, key=lambda g: g.centre):
        a, b = gap.centre - gap.width / 2, gap.centre + gap.width / 2
        if a < start - EPS or b > end + EPS:
            raise ValueError(f"gap at {gap.centre} (width {gap.width}) leaves [{start}, {end}]")
        if a - cursor > EPS:
            parts.append((cursor, a))
        cursor = b
    if end - cursor > EPS:
        parts.append((cursor, end))
    return parts


def wall_x(
    prefix: str,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    gaps: Sequence[Gap] = (),
    height: float = 2.0,
    role: Role = Role.WALL,
    material: str = "inner",
) -> list[Primitive]:
    """A wall along x occupying y ∈ [y0, y1], from x0 to x1, with door gaps cut out.

    Segments are named `<prefix>_00`, `<prefix>_01`, … from west to east.
    """
    parts = _segments(x0, x1, gaps)
    return [
        box(f"{prefix}_{i:02d}", a, b, y0, y1, height, role, material)
        for i, (a, b) in enumerate(parts)
    ]


def wall_y(
    prefix: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    gaps: Sequence[Gap] = (),
    height: float = 2.0,
    role: Role = Role.WALL,
    material: str = "inner",
) -> list[Primitive]:
    """A wall along y occupying x ∈ [x0, x1], from y0 to y1, with door gaps cut out (south to north)."""
    parts = _segments(y0, y1, gaps)
    return [
        box(f"{prefix}_{i:02d}", x0, x1, a, b, height, role, material)
        for i, (a, b) in enumerate(parts)
    ]


def straight_wall(
    prefix: str,
    side: str,
    inner: float,
    start: float,
    end: float,
    recesses: Iterable[Recess] = (),
    thickness: float = 0.2,
    height: float = 2.0,
    role: Role = Role.BOUNDARY_WALL,
    material: str = "boundary",
    gaps: Sequence[Gap] = (),
) -> tuple[list[Primitive], list[Notch]]:
    """One straight wall whose inner face lies on `inner`, bounding free space on its `side`.

    - `side` "s": the wall is the south side of a room, occupying y ∈ [inner − t, inner], x ∈ [start, end];
      "n": y ∈ [inner, inner + t]; "w": x ∈ [inner − t, inner], y ∈ [start, end]; "e": x ∈ [inner, inner + t].
    - Each recess replaces a wall section by a three-sided notch `depth` behind the inner face: a gap of the
      recess width, two side walls and a back wall, so the boundary stays closed. The notch is returned as a
      free-space landmark polygon.
    - `gaps` are plain door openings.
    """
    t = thickness
    items = list(recesses)
    cut = [Gap(r.centre, r.width) for r in items] + list(gaps)
    prims: list[Primitive] = []
    notches: list[Notch] = []
    if side in ("s", "n"):
        y0, y1 = (inner - t, inner) if side == "s" else (inner, inner + t)
        prims += wall_x(f"{prefix}_{side}", y0, y1, start, end, cut, height, role, material)
    elif side in ("w", "e"):
        x0, x1 = (inner - t, inner) if side == "w" else (inner, inner + t)
        prims += wall_y(f"{prefix}_{side}", x0, x1, start, end, cut, height, role, material)
    else:
        raise ValueError(f"unknown wall side {side!r}")
    for r in items:
        a, b, d = r.centre - r.width / 2, r.centre + r.width / 2, r.depth
        name = f"recess_{r.id.lower()}"
        if side == "s":
            prims += [
                box(f"{name}_a", a - t, a, inner - d - t, inner - t, height, role, material),
                box(f"{name}_b", b, b + t, inner - d - t, inner - t, height, role, material),
                box(f"{name}_back", a, b, inner - d - t, inner - d, height, role, material),
            ]
            poly = ((a, inner - d), (b, inner - d), (b, inner), (a, inner))
        elif side == "n":
            prims += [
                box(f"{name}_a", a - t, a, inner + t, inner + d + t, height, role, material),
                box(f"{name}_b", b, b + t, inner + t, inner + d + t, height, role, material),
                box(f"{name}_back", a, b, inner + d, inner + d + t, height, role, material),
            ]
            poly = ((a, inner), (b, inner), (b, inner + d), (a, inner + d))
        elif side == "w":
            prims += [
                box(f"{name}_a", inner - d - t, inner - t, a - t, a, height, role, material),
                box(f"{name}_b", inner - d - t, inner - t, b, b + t, height, role, material),
                box(f"{name}_back", inner - d - t, inner - d, a, b, height, role, material),
            ]
            poly = ((inner - d, a), (inner, a), (inner, b), (inner - d, b))
        else:
            prims += [
                box(f"{name}_a", inner + t, inner + d + t, a - t, a, height, role, material),
                box(f"{name}_b", inner + t, inner + d + t, b, b + t, height, role, material),
                box(f"{name}_back", inner + d, inner + d + t, a, b, height, role, material),
            ]
            poly = ((inner, a), (inner + d, a), (inner + d, b), (inner, b))
        notches.append(Notch(r.id, poly))
    return prims, notches


def walled_rectangle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    recesses: dict[str, Iterable[Recess]] | None = None,
    thickness: float = 0.2,
    height: float = 2.0,
    prefix: str = "outer",
) -> tuple[list[Primitive], list[Notch]]:
    """A closed outer wall whose inner faces lie on the rectangle [xmin, xmax] × [ymin, ymax].

    The south and north walls span the full outer length (xmin − t to xmax + t), so the corners overlap; the west
    and east walls span ymin to ymax. Recesses per side ("s", "n", "w", "e") as in `straight_wall`.
    """
    t = thickness
    rec = {side: list(items) for side, items in (recesses or {}).items()}
    for side in rec:
        if side not in ("s", "n", "w", "e"):
            raise ValueError(f"unknown wall side {side!r}")
    prims: list[Primitive] = []
    notch_by_side: dict[str, list[Notch]] = {}
    for side, inner, a0, a1 in (
        ("s", ymin, xmin - t, xmax + t),
        ("n", ymax, xmin - t, xmax + t),
        ("w", xmin, ymin, ymax),
        ("e", xmax, ymin, ymax),
    ):
        walls, notches = straight_wall(prefix, side, inner, a0, a1, rec.get(side, []), t, height)
        prims += walls
        notch_by_side[side] = notches
    return prims, [n for side in sorted(rec) for n in notch_by_side[side]]


def door(door_id: str, x: float, y: float, width: float, wall_axis: str) -> Door:
    """A doorway record for the E plans' doorway regions (T0 §4)."""
    if wall_axis not in ("x", "y"):
        raise ValueError("wall_axis must be 'x' or 'y'")
    return Door(door_id, x, y, width, wall_axis)


def grid_walls(
    occupancy: BoolArray,
    resolution: float,
    min_run: int = 1,
    origin: tuple[float, float] = (0.0, 0.0),
    prefix: str = "wall",
    height: float = 2.0,
    role: Role = Role.WALL,
    material: str = "inner",
) -> list[Primitive]:
    """Merged wall boxes covering an occupancy grid (row r spans y, column c spans x, cell (0, 0) at `origin`).

    Each row is split into runs of occupied cells along x (runs shorter than `min_run` are dropped); identical
    runs in consecutive rows merge into one box. The boxes are disjoint and named `<prefix>_000`, … in the order
    of their south-west corners (row, then column).
    """
    ox, oy = origin
    rows, cols = occupancy.shape
    found: list[tuple[int, int, int, int]] = []  # r0, r1 (exclusive), c0, c1 (exclusive)
    open_runs: dict[tuple[int, int], int] = {}
    for r in range(rows + 1):
        runs: list[tuple[int, int]] = []
        if r < rows:
            c = 0
            line = occupancy[r]
            while c < cols:
                if line[c]:
                    start = c
                    while c < cols and line[c]:
                        c += 1
                    if c - start >= min_run:
                        runs.append((start, c))
                else:
                    c += 1
        current = set(runs)
        for run in [k for k in open_runs if k not in current]:
            found.append((open_runs.pop(run), r, run[0], run[1]))
        for run in runs:
            open_runs.setdefault(run, r)
    found.sort()
    return [
        box(
            f"{prefix}_{i:03d}",
            ox + c0 * resolution,
            ox + c1 * resolution,
            oy + r0 * resolution,
            oy + r1 * resolution,
            height,
            role,
            material,
        )
        for i, (r0, r1, c0, c1) in enumerate(found)
    ]
